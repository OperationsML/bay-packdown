from typing import List, Tuple
import os
import logging

from commons import utils, bq_client
from google.cloud import bigquery
import datetime
import numpy as np
import math
import tensorflow as tf
from tensorflow.keras import mixed_precision
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm
import time
import pandas as pd

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
plt.style.use(f'{PARENT_DIR}/commons/custom_style.mplstyle')

log = logging.getLogger(__name__)
log.setLevel(os.environ.get("LOG_LEVEL", "INFO"))
date_ = datetime.datetime.now()

policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)
print('Compute dtype: %s' % policy.compute_dtype)
print('Variable dtype: %s' % policy.variable_dtype)

class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super(TransformerBlock, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate
        self.att = tf.keras.layers.MultiHeadAttention(num_heads=self.num_heads, key_dim=self.embed_dim)
        self.ffn = tf.keras.Sequential(
            [tf.keras.layers.Dense(self.ff_dim, activation="relu"), tf.keras.layers.Dense(self.embed_dim), ]
        )
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(self.rate)
        self.dropout2 = tf.keras.layers.Dropout(self.rate)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'ff_dim': self.ff_dim,
            'rate': self.rate,
            'att': self.att,
            'ffn': self.ffn,
            'layernorm1': self.layernorm1,
            'layernorm2': self.layernorm2,
            'dropout1': self.dropout1,
            'dropout2': self.dropout2
        })
        return config


    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)



class TokenAndPositionEmbedding(tf.keras.layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim, **kwargs):
        super(TokenAndPositionEmbedding, self).__init__()
        self.maxlen = maxlen
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.token_emb = tf.keras.layers.Embedding(input_dim=self.vocab_size, output_dim=self.embed_dim, mask_zero=True)
        self.pos_emb = tf.keras.layers.Embedding(input_dim=self.maxlen, output_dim=self.embed_dim, mask_zero=True)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'maxlen': self.maxlen,
            'vocab_size': self.vocab_size,
            'embed_dim': self.embed_dim,
            'token_emb': self.token_emb,
            'pos_emb': self.pos_emb
        })
        return config

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions


def tilted_loss(q, y, f):
    e = (y - f)
    max_ = tf.maximum(q * e, (q - 1) * e)
    if len(max_.shape) > 1:
        max_ = tf.squeeze(max_)

    return tf.reduce_mean(tf.squeeze(max_), axis=-1)

def binary_focal_loss_fixed(y_true, y_pred):
    """
    y_true shape need be (None,1)
    y_pred need be compute after sigmoid
    """
    gamma = tf.constant(2, dtype=tf.float32)
    alpha = tf.constant(.35, dtype=tf.float32)

    y_true = tf.cast(y_true, tf.float32)
    alpha_t = y_true * alpha + (tf.ones_like(y_true) - y_true) * (1 - alpha)

    p_t = y_true * y_pred + (tf.ones_like(y_true) - y_true) * (tf.ones_like(y_true) - y_pred) + tf.keras.backend.epsilon()
    focal_loss = - alpha_t * tf.pow((tf.ones_like(y_true) - p_t), gamma) * tf.math.log(p_t)

    return tf.reduce_mean(focal_loss)


class EarlyStopping():
    def __init__(self, patience, min_diff):
        self.best_weights = None
        self.patience = patience
        self.min_diff = min_diff
        self.best_epoch = 0
        self.best = np.Inf
        self.wait = 0
        self.stop_training = False

    def on_epoch_end(self, model, current_test_val, epoch):
        stop_training = False
        if epoch == 1:
            self.best = current_test_val
            self.best_weights = model.get_weights()
            self.best_epoch = epoch
        else:
            if np.less(current_test_val, self.best):
                self.best_weights = model.get_weights()
                self.best_epoch = epoch

                if (self.best - current_test_val) < self.min_diff:
                    self.wait += 1

                else:
                    self.best = current_test_val
                    self.wait = 0

                if self.wait >= self.patience:
                    stop_training = True
                else:
                    stop_training = False

            else:
                self.wait += 1
                if self.wait >= self.patience:
                    stop_training = True

        return stop_training, self.best

    def on_train_end(self, model):
        print('End of training')
        print(f'Best val: {self.best}, Best epoch: {self.best_epoch}, patience:{self.wait}')
        model.set_weights(self.best_weights)

        return model

class monitor_metric():
    def __init__(self, fn):
        self.y = np.asanyarray([]).reshape(-1,1)
        self.preds = np.asanyarray([]).reshape(-1,1)
        self.fn = fn

    def update_state(self, y, preds):
        self.y = np.append(self.y, np.array(y), axis=0)
        self.preds = np.append(self.preds, np.array(preds), axis=0)

    def result(self):
        result = self.fn(self.y, self.preds)
        return result

    def reset_states(self):
        self.y = np.asanyarray([]).reshape(-1,1)
        self.preds = np.asanyarray([]).reshape(-1,1)

def nn_model(num_data_shape, maxlen, vocab_size):
    # sku desc
    embed_dim = 100
    num_heads = 1
    hidden_dim = 128

    attribute = 'SKU_TOKENS'
    sku_inputs = tf.keras.layers.Input(shape=(maxlen,), name=attribute)
    embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size+1, embed_dim)
    sku_x = embedding_layer(sku_inputs)
    sku_transformer_block = TransformerBlock(embed_dim, num_heads, hidden_dim)
    sku_x = sku_transformer_block(sku_x)
    sku_x = tf.keras.layers.GlobalAveragePooling1D()(sku_x)
    sku_x = tf.keras.layers.Dropout(0.1)(sku_x)
    sku_x = tf.keras.layers.Dense(16, activation="relu")(sku_x)

    num_data_inputs = tf.keras.layers.Input(shape=(num_data_shape,), name='NUM_DATA')
    num_data_x = tf.keras.layers.concatenate([num_data_inputs, sku_x])

    x = tf.keras.layers.Dense(256, activation='relu')(num_data_x)
    stage01_x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    x = tf.keras.layers.Dense(16, activation='relu')(x)

    oos = tf.keras.layers.Dense(1, activation='linear', name='OOS_PERC', dtype='float32')(x)
    out = tf.keras.layers.Dense(1, activation='linear', name='SHELF_OUT', dtype='float32')(x)
    low = tf.keras.layers.Dense(1, activation='linear', name='SHELF_LOW', dtype='float32')(x)

    all_data = tf.keras.layers.concatenate([stage01_x, oos, out, low])
    x = tf.keras.layers.Dense(64, activation='relu')(all_data)
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    x = tf.keras.layers.Dense(16, activation='relu')(x)
    x = tf.keras.layers.Dense(8, activation='relu')(x)

    retl_opp_ind = tf. keras.layers.Dense(1, activation='sigmoid', name='BAY_RETL_OPP_IND', dtype='float32')(x)

    output_dict = {}
    output_dict['opp_1'] = tf.keras.layers.Dense(1, activation='linear', name='BAY_RETL_OPP_1', dtype='float32')(x)
    x = tf.keras.layers.concatenate([all_data, retl_opp_ind, output_dict['opp_1']])
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
    x = tf.keras.layers.Dense(8, activation='relu')(x)
    output_dict['opp_2'] = tf.keras.layers.Dense(1, activation='linear', name='BAY_RETL_OPP_2', dtype='float32')(x)
    dict_outs = [value for key, value in output_dict.items()]

    model = tf.keras.Model(
        inputs={'NUM_DATA': num_data_inputs,
                'SKU_TOKENS': sku_inputs},
        outputs=[oos, out, low, retl_opp_ind, *dict_outs]
    )
    return model



def net_training_fn(train_data: Tuple,
                    test_data: Tuple,
                    eval_data: Tuple,
                    model_name: str,
                    batch_size: int,
                    epochs: int,
                    metadata_table: str,
                    data_features: List[str],
                    exp_name: str,
                    maxlen: int,
                    vocab_size: int,
                    ):

    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    gpus = tf.config.list_physical_devices('GPU')
    print(gpus)
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    x_train, train_tokens, y_train = train_data
    x_test, test_tokens, y_test = test_data
    x_eval, eval_tokens, y_eval = eval_data

    # create tf. dataset
    train_out_dict = {'OOS_PERC': y_train['OOS_PERC'].to_numpy().reshape(-1,1),
                      'SHELF_OUT': y_train['SHELF_OUT'].to_numpy().reshape(-1,1),
                      'SHELF_LOW': y_train['SHELF_LOW'].to_numpy().reshape(-1, 1),
                      'BAY_RETL_OPP_IND': y_train['RETL_OPP_BINARY'].to_numpy().reshape(-1, 1),
                      'BAY_RETL_OPP': y_train['BAY_LOG_RETL_OPP'].to_numpy().reshape(-1, 1)
                      }

    test_out_dict = {'OOS_PERC': y_test['OOS_PERC'].to_numpy().reshape(-1,1),
                      'SHELF_OUT': y_test['SHELF_OUT'].to_numpy().reshape(-1,1),
                      'SHELF_LOW': y_test['SHELF_LOW'].to_numpy().reshape(-1, 1),
                      'BAY_RETL_OPP_IND': y_test['RETL_OPP_BINARY'].to_numpy().reshape(-1, 1),
                      'BAY_RETL_OPP': y_test['BAY_LOG_RETL_OPP'].to_numpy().reshape(-1, 1)
                      }

    eval_out_dict = {'OOS_PERC': y_eval['OOS_PERC'].to_numpy().reshape(-1,1),
                     'SHELF_OUT': y_eval['SHELF_OUT'].to_numpy().reshape(-1,1),
                     'SHELF_LOW': y_eval['SHELF_LOW'].to_numpy().reshape(-1, 1),
                     'BAY_RETL_OPP_IND': y_eval['RETL_OPP_BINARY'].to_numpy().reshape(-1, 1),
                     'BAY_RETL_OPP': y_eval['BAY_LOG_RETL_OPP'].to_numpy().reshape(-1, 1)
                      }

    train_data = tf.data.Dataset.from_tensor_slices(({'NUM_DATA': x_train, 'SKU_TOKENS': train_tokens}, train_out_dict))
    test_data = tf.data.Dataset.from_tensor_slices(({'NUM_DATA': x_test, 'SKU_TOKENS': test_tokens}, test_out_dict))
    eval_data = tf.data.Dataset.from_tensor_slices(({'NUM_DATA': x_eval, 'SKU_TOKENS': eval_tokens}, eval_out_dict))

    steps_per_epoch = x_train.shape[0] // batch_size
    train_ds = train_data.shuffle(x_train.shape[0]).batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
    test_ds = test_data.shuffle(x_train.shape[0]).batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
    eval_ds = eval_data.shuffle(x_train.shape[0]).batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

    model = nn_model(x_train.shape[1], maxlen, vocab_size)
    plot_model(model, to_file='model.png')
    print(model.summary())

    optimizer = tf.keras.optimizers.SGD(learning_rate=1e-2, decay=0.01, momentum=0.9, nesterov=True)
    loss_50th_fn = lambda y, f: tilted_loss(0.50, y, f)
    loss_opp_50th_fn = lambda y, f: tilted_loss(0.50, y, f)

    # define losses for outputs
    train_oos_metric_fn = monitor_metric(loss_50th_fn)
    train_out_metric_fn = monitor_metric(loss_50th_fn)
    train_low_metric_fn = monitor_metric(loss_50th_fn)
    train_opp_ind_precision_fn = tf.keras.metrics.Precision()
    train_opp_ind_recall_fn = tf.keras.metrics.Recall()
    train_opp_val_metric_fn = monitor_metric(loss_opp_50th_fn)

    test_out_metric_fn = monitor_metric(loss_50th_fn)
    test_opp_ind_precision_fn = tf.keras.metrics.Precision()
    test_opp_ind_recall_fn = tf.keras.metrics.Recall()
    test_opp_val_metric_fn = monitor_metric(loss_opp_50th_fn)

    early_stopping = EarlyStopping(5, .01)


    @tf.function
    def train_step(x, y):
        with tf.GradientTape() as tape:
            preds = model(x, training=True)
            # Non-zero indices
            non_zero_idx = tf.not_equal(y['BAY_RETL_OPP'], 0)
            y_opp = y['BAY_RETL_OPP'][non_zero_idx]
            # bay level percentages
            q50_oos_loss = loss_50th_fn(y['OOS_PERC'], preds[0])
            q50_out_loss = loss_50th_fn(y['SHELF_OUT'], preds[1])
            q50_low_loss = loss_50th_fn(y['SHELF_LOW'], preds[2])
            # binary indicator
            retl_opp_ind_loss = binary_focal_loss_fixed(y['BAY_RETL_OPP_IND'], preds[3])
            opp_losses = [loss_opp_50th_fn(y_opp, preds[i][non_zero_idx]) for i in range(4, len(preds))]
        grads = tape.gradient([q50_oos_loss, q50_out_loss, q50_low_loss, retl_opp_ind_loss, *opp_losses], model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

        return preds[0], preds[1], preds[2], preds[3], preds[-1]

    @tf.function
    def test_step(x,y):
        preds = model(x, training=False)
        return preds[0], preds[1], preds[2], preds[3], preds[-1]

    #Train
    history={'epoch': [],
             'train_loss': [],
             'test_loss': [],
             }
    for epoch in range(1, epochs + 1):
        print(f'\nStart of epoch: {epoch}')
        start_time = time.time()

        for step, (x_batch_train, y_batch_train) in zip(tqdm(range(steps_per_epoch), position=0, leave=True), train_ds):
            q50_oos_pred, q50_out_pred, q50_low_pred, retl_opp_ind_pred, q50_opp_val_pred = train_step(x_batch_train, y_batch_train)

            # Non-zero  RETL_OPP indices
            non_zero_idx = tf.not_equal(y_batch_train['BAY_RETL_OPP'], 0)
            y_batch_opp = tf.reshape(y_batch_train['BAY_RETL_OPP'][non_zero_idx], [-1,1])
            q50_opp_val_pred = tf.reshape(q50_opp_val_pred[non_zero_idx], [-1,1])
            train_opp_val_metric_fn.update_state(y_batch_opp, q50_opp_val_pred)

            # OTHER DATA
            train_oos_metric_fn.update_state(y_batch_train['OOS_PERC'], q50_oos_pred)
            train_out_metric_fn.update_state(y_batch_train['SHELF_OUT'], q50_out_pred)
            train_low_metric_fn.update_state(y_batch_train['SHELF_LOW'], q50_low_pred)

            train_opp_ind_precision_fn.update_state(y_batch_train['BAY_RETL_OPP_IND'], retl_opp_ind_pred)
            train_opp_ind_recall_fn.update_state(y_batch_train['BAY_RETL_OPP_IND'], retl_opp_ind_pred)


        metrics = []
        for label_, metric_fn in zip(['oos', 'out', 'low', 'opp_val', 'retl_opp_ind_precision', 'retl_opp_ind_recall'], [train_oos_metric_fn, train_out_metric_fn, train_low_metric_fn, train_opp_val_metric_fn, train_opp_ind_precision_fn, train_opp_ind_recall_fn]):
            metric = round(float(metric_fn.result()), 4)
            metrics.append(f'{label_}: {metric}')
            metric_fn.reset_states()
            if label_ == 'out':
                history['epoch'].append(epoch)
                history['train_loss'].append(metric)
        print(f'Training metrics over epoch:', sep='')
        print(*metrics, sep=' ')

        for x_batch_test, y_batch_test in test_ds:
            _, q50_out_pred, _, retl_opp_ind_pred, q50_opp_val_pred = test_step(x_batch_test, y_batch_test)
            # Non-zero indices
            non_zero_idx = tf.not_equal(y_batch_test['BAY_RETL_OPP'], 0)
            y_batch_opp = tf.reshape(y_batch_test['BAY_RETL_OPP'][non_zero_idx], [-1, 1])
            q50_opp_val_pred = tf.reshape(q50_opp_val_pred[non_zero_idx], [-1, 1])

            test_opp_ind_precision_fn.update_state(y_batch_test['BAY_RETL_OPP_IND'], retl_opp_ind_pred)
            test_opp_ind_recall_fn.update_state(y_batch_test['BAY_RETL_OPP_IND'], retl_opp_ind_pred)

            test_out_metric_fn.update_state(y_batch_test['SHELF_OUT'], q50_out_pred)
            test_opp_val_metric_fn.update_state(y_batch_opp, q50_opp_val_pred)

        test_out_metric = round(float(test_out_metric_fn.result()), 4)
        test_opp_precision_metric = round(float(test_opp_ind_precision_fn.result()), 4)
        test_opp_recall_metric = round(float(test_opp_ind_recall_fn.result()), 4)
        test_opp_val_metric = round(float(test_opp_val_metric_fn.result()), 4)

        test_opp_val_metric_fn.reset_states()
        test_out_metric_fn.reset_states()
        test_opp_ind_precision_fn.reset_states()
        test_opp_ind_recall_fn.reset_states()

        history['test_loss'].append(test_out_metric)
        print(f'Test out q50: {float(test_out_metric)}')
        print(f'Test opp val q50: {float(test_opp_val_metric)}')
        print(f'Test opp ind precision: {float(test_opp_precision_metric)}')
        print(f'Test opp ind recall: {float(test_opp_recall_metric)}')
        print(f'Time taken: {time.time() - start_time}')

        stop_training, best_val = early_stopping.on_epoch_end(model, test_opp_val_metric, epoch)
        print(f'Best Val: {best_val}')
        if stop_training:
            model = early_stopping.on_train_end(model)
            break
        else:
            continue

    if not stop_training:
        model = early_stopping.on_train_end(model)

    #Evaluation
    for x_batch_test, y_batch_test in eval_ds:
        _, q50_out_pred, _, retl_opp_ind_pred, q50_opp_val_pred = test_step(x_batch_test, y_batch_test)

        # Non-zero indices
        non_zero_idx = tf.not_equal(y_batch_test['BAY_RETL_OPP'], 0)
        y_batch_opp = tf.reshape(y_batch_test['BAY_RETL_OPP'][non_zero_idx], [-1, 1])
        q50_opp_val_pred = tf.reshape(q50_opp_val_pred[non_zero_idx], [-1, 1])

        test_out_metric_fn.update_state(y_batch_test['SHELF_OUT'], q50_out_pred)
        test_opp_ind_precision_fn.update_state(y_batch_test['BAY_RETL_OPP_IND'], retl_opp_ind_pred)
        test_opp_ind_recall_fn.update_state(y_batch_test['BAY_RETL_OPP_IND'], retl_opp_ind_pred)
        test_opp_val_metric_fn.update_state(y_batch_opp, q50_opp_val_pred)

    eval_out_metric = round(float(test_out_metric_fn.result()), 4)
    eval_opp_val_metric = round(float(test_opp_val_metric_fn.result()), 4)
    test_opp_precision_metric = round(float(test_opp_ind_precision_fn.result()), 4)
    test_opp_recall_metric = round(float(test_opp_ind_recall_fn.result()), 4)
    print(f'Eval out q50: {float(eval_out_metric)}')
    print(f'Eval opp val q50: {float(eval_opp_val_metric)}')
    print(f'Test opp ind precision: {float(test_opp_precision_metric)}')
    print(f'Test opp ind recall: {float(test_opp_recall_metric)}')

    #Save model
    gcs_path = f'gs://store-ops-ml/artifacts/{exp_name}'
    weight_path = f'{gcs_path}/{model_name}/checkpoint'
    model.save_weights(weight_path, save_format='tf')

    model_config = model.get_config()
    config_path = utils.save_model(model_config, f'{model_name}_config', 'store-ops-ml', exp_name)

    print(weight_path, gcs_path)

    results = pd.DataFrame.from_dict(
        dict(DATE=date_,
             MODEL_TYPE=['quantile_regression'],
             MODEL_LABEL=f"multi-output",
             VERSION=f"{model_name}_{exp_name}",
             DATA_FEATURES=[f"{data_features}" or None],
             EVAL_OUT=eval_out_metric,
             EVAL_OPP=eval_opp_val_metric,
             MODEL_LOCATION=[gcs_path],
             CONFIG_PATH=[config_path]
             )
    )

    job_config = bigquery.LoadJobConfig(
        schema=[
            bigquery.SchemaField("MODEL_TYPE", bigquery.enums.SqlTypeNames.STRING),
            bigquery.SchemaField("MODEL_LABEL", bigquery.enums.SqlTypeNames.STRING),
            bigquery.SchemaField("VERSION", bigquery.enums.SqlTypeNames.STRING),
            bigquery.SchemaField("DATA_FEATURES", bigquery.enums.SqlTypeNames.STRING),
            bigquery.SchemaField("EVAL_OUT", bigquery.enums.SqlTypeNames.FLOAT64),
            bigquery.SchemaField("EVAL_OPP", bigquery.enums.SqlTypeNames.FLOAT64),
            bigquery.SchemaField("MODEL_LOCATION", bigquery.enums.SqlTypeNames.STRING),
            bigquery.SchemaField("CONFIG_PATH", bigquery.enums.SqlTypeNames.STRING),
        ],
        # write_disposition="WRITE_TRUNCATE",
    )

    job = bq_client.get_client().load_table_from_dataframe(results, metadata_table, job_config=job_config)
    job.result()
    print(f'Uploaded results to: `{metadata_table}`')


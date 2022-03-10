from typing import List

import os
import logging

from commons import utils, bq_client
from inference.nn_model_training import net_training_fn
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
import optuna
import lightgbm as lgb
import tensorflow as tf
import datetime
import numpy as np
import pandas as pd
import math

log = logging.getLogger(__name__)
log.setLevel(os.environ.get("LOG_LEVEL", "INFO"))

date_ = datetime.datetime.now()


def training_fn(bq_project: str,
                bq_dataset: str,
                train_table_id: str,
                model_name: str,
                metadata_table_id: str,
                filter_list: List[str],
                git_branch: str,
                batch_size: int,
                epochs: int,
                data_limit: int = None,
                vocab_size: int = 20000,
                labels: List[str] = None,
                hyperparameter_tuning=True):
    """Runs training on a model against the given data from BigQuery

    :param bq_project: BigQuery project ID where the dataset is residing
    :param bq_dataset: BigQuery Dataset ID where the the table is residing
    :param embedding_features: List of features for entity embeddings
    :param metadata_table_id: Table ID that holds training metadata
    :param filter_list: List of columns to filter from data
    :param git_branch: git branch to associated with code
    :param data_limit: number of records to run training on
    :param regression_labels: List of labels used for regression targets
    :param classification_labels: List of labels used for classification targets
    :param hyperparameter_tuning: Boolean that indicates if hyperparameter tuning will be done

    """

    print(f"Starting training: "
          f"| project: {bq_project} "
          f"| dataset: {bq_dataset} "
          f"| table: {train_table_id}")

    exp_name = str(math.floor(datetime.datetime.now().timestamp())) + f'_{git_branch}'
    print(f'Experiment: {exp_name}')

    train_table = f'{bq_project}.{bq_dataset}.{train_table_id}'
    metadata_table = f'{bq_project}.{bq_dataset}.{metadata_table_id}_{git_branch}'

    # Pull data
    data = utils.query_data(train_table, reduce_mem=True, predict_data=False, limit=data_limit)

    # Get features and type
    features = utils.get_features(train_table, filter_list)
    categorical_cols, numeric_cols = utils.split_column_type(features, exclude_cols=labels)

    print(f'categorical features: {categorical_cols}')
    print(f'numerical features: {numeric_cols}')

    # column transformer
    num_pipe = Pipeline([('imputer', SimpleImputer()), ('normalize', StandardScaler())])
    transformer = ColumnTransformer(transformers=[('num', num_pipe, numeric_cols),
                                                  ('cat', OneHotEncoder(handle_unknown='ignore', sparse=False,
                                                                        dtype=np.float32), categorical_cols),
                                                  ],
                                    # one hot encode all categoricals
                                    sparse_threshold=0,
                                    remainder='passthrough'
                                    )

    print(f"Labels: {labels}")

    x_train, y_train = utils.preprocess(data.loc[data['DATA_LABEL'] == 'TRAIN'],
                                        numeric_cols,
                                        categorical_cols,
                                        label=labels,
                                        dense=True,
                                        transformer=transformer,
                                        train=True)

    x_test, y_test = utils.preprocess(data.loc[data['DATA_LABEL'] == 'TEST'],
                                      numeric_cols,
                                      categorical_cols,
                                      label=labels,
                                      dense=True,
                                      transformer=transformer,
                                      train=False)

    x_eval, y_eval = utils.preprocess(data.loc[data['DATA_LABEL'] == 'EVAL'],
                                      numeric_cols,
                                      categorical_cols,
                                      label=labels,
                                      dense=True,
                                      transformer=transformer,
                                      train=False)

    # tokenize data
    tokenizer_dict = {}
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=vocab_size, oov_token="<unk>")
    tokenizer.fit_on_texts(data.loc[data['DATA_LABEL'] == 'TRAIN']['SKUS'])
    tokenizer.word_index['<pad>'] = 0
    tokenizer.index_word[0] = '<pad>'
    tokenizer_dict['tokenizer'] = tokenizer

    # Create the tokenized vectors
    train_tokens = tokenizer.texts_to_sequences(data.loc[data['DATA_LABEL'] == 'TRAIN']['SKUS'])
    train_tokens = tf.keras.preprocessing.sequence.pad_sequences(train_tokens, padding='post')
    tokenizer_dict['maxlen'] = train_tokens.shape[1]

    test_tokens = tokenizer.texts_to_sequences(data.loc[data['DATA_LABEL'] == 'TEST']['SKUS'])
    test_tokens = tf.keras.preprocessing.sequence.pad_sequences(test_tokens, padding='post',
                                                                maxlen=tokenizer_dict['maxlen'])

    eval_tokens = tokenizer.texts_to_sequences(data.loc[data['DATA_LABEL'] == 'EVAL']['SKUS'])
    eval_tokens = tf.keras.preprocessing.sequence.pad_sequences(eval_tokens, padding='post',
                                                                maxlen=tokenizer_dict['maxlen'])

    ohe_path = utils.save_model(transformer, f'{model_name}_transformer', 'store-ops-ml', exp_name)
    tokenizer_path = utils.save_model(tokenizer_dict, f'{model_name}_tokenizer', 'store-ops-ml', exp_name)

    print(f'Train shape: {x_train.shape}, {train_tokens.shape}, {y_train.shape}')
    print(f'Test shape: {x_test.shape}, {test_tokens.shape}, {y_test.shape}')
    print(f'Eval shape: {x_eval.shape}, {eval_tokens.shape}, {y_eval.shape}')
    feature_names = utils.get_feature_names(transformer)

    net_training_fn(train_data=(x_train, train_tokens, y_train),
                    test_data=(x_test, test_tokens, y_test),
                    eval_data=(x_eval, eval_tokens, y_eval),
                    data_features=[*categorical_cols, *numeric_cols],
                    model_name=model_name,
                    batch_size=batch_size,
                    epochs=epochs,
                    metadata_table=metadata_table,
                    exp_name=exp_name,
                    maxlen=tokenizer_dict['maxlen'],
                    vocab_size=vocab_size
                    )

from commons import utils, bq_client
from typing import List, Callable, Dict
import joblib
import numpy as np
import pandas as pd
import datetime
import tensorflow as tf
from tqdm import tqdm

def inference_reg_fn(data_dict: Dict,
                     bay_str_data: pd.DataFrame,
                     prediction_output_table: str,
                     model: Callable,
                     batch_size: int = 256,
                     ):
    """Runs inference on a model against the given data from BigQuery

     :param batch_size: Batch size for inference
     :param predict_data: Data for prediction
     :param bay_str_data: Name of gcs bucket
     :param prediction_output_table: Table to save results to
     :param model: model for prediction
     """

    pred_data = tf.data.Dataset.from_tensor_slices({
                                                    'NUM_DATA': data_dict['NUM_DATA'],
                                                    'SKU_TOKENS': data_dict['SKU_TOKENS']
                                                    })

    pred_data = pred_data.batch(batch_size, drop_remainder=False).prefetch(1)
    pred_keys = ['oos', 'out', 'low', 'opp_bin_ind', 'opp_val']
    preds_dict = {**dict.fromkeys(pred_keys, np.array([]))}

    steps_per_epoch = data_dict['NUM_DATA'].shape[0] // batch_size
    for step, x_batch in zip(tqdm(range(steps_per_epoch+1), position=0, leave=True), pred_data):
        preds = model(x_batch, training=False)
        preds_dict['oos'] = np.append(preds_dict['oos'], preds[0].numpy().flatten(), axis=0)
        preds_dict['out'] = np.append(preds_dict['out'], preds[1].numpy().flatten(), axis=0)
        preds_dict['low'] = np.append(preds_dict['low'], preds[2].numpy().flatten(), axis=0)
        preds_dict['opp_bin_ind'] = np.append(preds_dict['opp_bin_ind'], preds[3].numpy().flatten(), axis=0)
        preds_dict['opp_val'] = np.append(preds_dict['opp_val'], preds[-1].numpy().flatten(), axis=0)

    bay_str_data = bay_str_data.copy().reset_index()
    print(bay_str_data.shape)
    for key, value in preds_dict.items():
        if key == 'opp_val':
            bay_str_data[f'{key.upper()}_PREDICTION'] = preds_dict[key]
        bay_str_data[f'{key.upper()}_PREDICTION'] = preds_dict[key]
    bay_str_data['DATE'] = datetime.datetime.now()

    # Create results data
    job = bq_client.get_client().load_table_from_dataframe(bay_str_data, prediction_output_table)
    job.result()



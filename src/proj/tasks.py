from .celery import app, Task
import redis
import logging
import pandas as pd
import sqlite3
import pickle
import numpy as np
from typing import List
import spectral.io.envi as envi
import random
import time


logging.basicConfig(level=logging.INFO)



def load_sklearn_model_to_file(file_path):
    with open(file_path, "rb") as file:
        return pickle.load(file)
    
    
def read_image_spectral():
    return envi.open(
        "/home/ubuntu/code/agriculture-prediction/data/spectral_image/hyper_20220913_3cm.hdr", 
        "/home/ubuntu/code/agriculture-prediction/data/spectral_image/hyper_20220913_3cm.img"
    )
    
def read_image_spectral_1():
    return envi.open(
        "/home/ubuntu/code/agriculture-prediction/data/spectral_image_1/hyper_20220913_3cm.hdr", 
        "/home/ubuntu/code/agriculture-prediction/data/spectral_image_1/hyper_20220913_3cm.img"
    )
    
def read_image_spectral_2():
    return envi.open(
        "/home/ubuntu/code/agriculture-prediction/data/spectral_image_2/hyper_20220913_3cm.hdr", 
        "/home/ubuntu/code/agriculture-prediction/data/spectral_image_2/hyper_20220913_3cm.img"
    )

def create_sqlite3_conn_pool(num_partition):
    pool = {}
    for partition in range(num_partition):
        pool[partition] = sqlite3.connect(f"./proj/db/agriculture_{partition}.db")
    return pool
    
NUM_PARTITION = 5
SQLITE3_CONN_POOL = create_sqlite3_conn_pool(NUM_PARTITION)
clf_RF_1 = load_sklearn_model_to_file("/home/ubuntu/code/agriculture-prediction/RF_save/clf_RF_1.pkl")
IMG = read_image_spectral()

def get_image_data(img, start_row, end_row, col, num_band):
    alternative_images = [
        read_image_spectral_1,
        read_image_spectral_2
    ]
    
    
    try:
        return img[start_row:end_row, col, :].reshape(end_row - start_row, num_band), 0
    except EOFError as err:
        img = alternative_images[0]()
        return img[start_row:end_row, col, :].reshape(end_row - start_row, num_band), 1
    except Exception as err:
        img = alternative_images[1]()
        return img[start_row:end_row, col, :].reshape(end_row - start_row, num_band), 2
        

def calulate_N(
    run_id: int,
    img: List,
    col: int,
    start_row: int,
    end_row: int,
    ):
    
    start_time = time.time()
    num_row, num_col, num_band = img.shape
    
    matrix, ix_folder = get_image_data(img, start_row, end_row, col, num_band)
    
    end_get_img_data = time.time()
    logging.info(f"Finished get img data of col {col}, folder = {ix_folder} in:     {end_get_img_data - start_time:.2f} seconds.")
    
    nitos = clf_RF_1.predict(matrix).tolist()
    
    result = pd.DataFrame({
        "col": col,
        "start_row": [start_row],
        "end_row": [end_row],
        "len_result": [len(nitos)],
        "run_id": [run_id],
        "nitos": [str(nitos)],
    })
    
    end_calculate_time = time.time()
    logging.info(f"Finished calculate in:                                {end_calculate_time - end_get_img_data:.2f} seconds.")
    
    partition = col % NUM_PARTITION
    sqlite3_conn = SQLITE3_CONN_POOL[partition]
    result.to_sql(
        f"nito_{run_id}",
        sqlite3_conn,
        if_exists="append",
        index=False
    )
    
    end_insert_to_sql_time = time.time()
    logging.info(f"Finished insert to sql in:                            {end_insert_to_sql_time - end_calculate_time:.2f} seconds.")
    logging.info(f"TOTAL:                                                {end_insert_to_sql_time - start_time:.2f} seconds.")
    


@app.task(bind=True)
def calculate_N_task( 
    self: Task,
    run_id: int, 
    col: int, 
    start_row: int, 
    end_row: int,
    ):
    
    err = None
    try:
        calulate_N(
            run_id,
            IMG,
            col,
            start_row,
            end_row
        )
    except Exception as e:
        err = e
    
    if err is None:
        return

    if type(err) == EOFError:
        time_to_retry = random.randint(0, 10)
        logging.info(f"End of file error: retry forever {time_to_retry} seconds")
        raise self.retry(countdown=time_to_retry, exc=err, max_retries=None, args=(run_id, col, start_row, end_row))
    else:
        logging.info(f"Error: {str(err)}")
        raise self.retry(countdown=10, exc=err, max_retries=3, args=(run_id, col, start_row, end_row))

    
def send_matrix_to_queue(
    run_id: int,
    img: List, 
    col: int, 
    chunk: int
    ):
    
    num_row, num_col, num_band = img.shape 
    
    start_row = 0
    while start_row < num_row:
        end_row = min(start_row + chunk, num_row)
        calculate_N_task.delay(
            run_id, 
            col, 
            start_row, 
            end_row,
        )
        start_row = end_row


@app.task(bind=True)
def send_matrix_to_queue_task(
    self: Task, 
    run_id: int,
    col: int, 
    chunk: int
    ):
    try:
        send_matrix_to_queue(
            run_id,
            IMG, 
            col, 
            chunk
        )
    except Exception as e:
        logging.info(f"Error: {str(e)}")
        raise self.retry(countdown=10, exc=e, max_retries=3)

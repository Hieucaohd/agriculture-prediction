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
import redis
import os


logging.basicConfig(level=logging.INFO)


def load_sklearn_model_to_file(file_path):
    with open(file_path, "rb") as file:
        return pickle.load(file)
    

def get_full_path(path):
    script_dir = os.path.dirname(os.path.realpath(__file__))
    return os.path.abspath(os.path.join(script_dir, path))
    
    
def read_image_spectral():
    return envi.open(
        get_full_path("../../data/spectral_image/hyper_20220913_3cm.hdr"), 
        get_full_path("../../data/spectral_image/hyper_20220913_3cm.img")
    )
    
def read_image_spectral_1():
    return envi.open(
        get_full_path("../../data/spectral_image_1/hyper_20220913_3cm.hdr"), 
        get_full_path("../../data/spectral_image_1/hyper_20220913_3cm.img")
    )
    
def read_image_spectral_2():
    return envi.open(
        get_full_path("../../data/spectral_image_2/hyper_20220913_3cm.hdr"), 
        get_full_path("../../data/spectral_image_2/hyper_20220913_3cm.img")
    )

def create_sqlite3_conn_pool(num_partition):
    pool = {}
    for partition in range(num_partition):
        pool[partition] = sqlite3.connect(get_full_path(f"./db/agriculture_{partition}.db"))
    return pool
    
    
NUM_PARTITION = 5
SQLITE3_CONN_POOL = create_sqlite3_conn_pool(NUM_PARTITION)
clf_RF_1 = load_sklearn_model_to_file(get_full_path("../../model_saved/RF_save/clf_RF_1.pkl"))
IMG = read_image_spectral()
REDIS_CONN = redis.Redis("localhost", 6379, db=6)


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


def get_image_data_from_redis(img, start_row, end_row, col, num_band):
    img_data_str = REDIS_CONN.get(f"{col}-{start_row}-{end_row}")
    
    if img_data_str is None:
        return get_image_data(img, start_row, end_row, col, num_band)
    
    img_data_list = eval(img_data_str)
    img_data_np = np.array(img_data_list)
    return img_data_np, "redis"


def read_col_data_from_file(file_path):
    with np.load(file_path) as file:
        return file["arr_0"]


def calculate_N_using_col_data(
    run_id: int, 
    col: int,
    ):
    start_time = time.time()
    
    file_path = get_full_path(f"../data/img_col_data/img_{col}.npz")
    matrix = read_col_data_from_file(file_path)
    
    end_get_img_data = time.time()
    logging.info(f"Finished get img data in:  {end_get_img_data - start_time:.2f} seconds. Col = {col}, file = {file_path}.")
    
    nitos = clf_RF_1.predict(matrix).tolist()
    
    result = pd.DataFrame({
        "col": col,
        "len_result": [len(nitos)],
        "run_id": [run_id],
        "nitos": [str(nitos)],
    })
    
    end_calculate_time = time.time()
    logging.info(f"Finished calculate in:     {end_calculate_time - end_get_img_data:.2f} seconds.")
    
    partition = col % NUM_PARTITION
    sqlite3_conn = SQLITE3_CONN_POOL[partition]
    table_name = f"nito_{run_id}"
    result.to_sql(
        table_name,
        sqlite3_conn,
        if_exists="append",
        index=False
    )
    
    end_insert_to_sql_time = time.time()
    logging.info(f"Finished insert to sql in: {end_insert_to_sql_time - end_calculate_time:.2f} seconds. Table name = {table_name}, partition = {partition}.")
    logging.info(f"TOTAL:                     {end_insert_to_sql_time - start_time:.2f} seconds.")


@app.task(bind=True)
def calculate_N_using_col_data_task(
    self: Task,
    run_id: int,
    col: int,
    ):
    try:
        calculate_N_using_col_data(
            run_id,
            col
        )
    except Exception as e:
        logging.info(f"Error: {str(e)}")
        raise self.retry(countdown=10, exc=e, max_retries=3, args=(run_id, col))


def calculate_N(
    run_id: int,
    img: List,
    col: int,
    start_row: int,
    end_row: int,
    ):
    
    start_time = time.time()
    num_row, num_col, num_band = img.shape
    
    matrix, ix_folder = get_image_data_from_redis(img, start_row, end_row, col, num_band)
    
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
        calculate_N(
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
        raise self.retry(countdown=10, exc=e, max_retries=3, args=(run_id, col, chunk))
    

def push_to_redis(
    col: int,
    start_row: int,
    end_row: int, 
    img_data: str,
    redis_conn: redis.Redis
    ):
    redis_conn.set(f"{col}-{start_row}-{end_row}", img_data)


def push_img_data_to_redis(
    img, 
    col: int, 
    start_row: int, 
    end_row: int, 
    redis_conn: redis.Redis
    ):
    num_row, num_col, num_band = img.shape
    
    start_time = time.time()
    
    img_data, _ = get_image_data(img, start_row, end_row, col, num_band)
    
    end_read_img_time = time.time()
    logging.info(f"Finished read img data in: {end_read_img_time - start_time} seconds.")
    
    img_data_list = img_data.tolist()
    push_to_redis(col, start_row, end_row, str(img_data_list), redis_conn)
    
    end_push_redis_time = time.time()
    logging.info(f"Finished push to redis in: {end_push_redis_time - end_read_img_time} seconds.")
    logging.info(f"TOTAL: {end_push_redis_time - start_time}")
    

@app.task(bind=True)
def push_img_data_to_redis_task(
    self: Task,
    col: int,
    start_row: int,
    end_row: int
    ):
    try:
        push_img_data_to_redis(
            IMG,
            col,
            start_row,
            end_row,
            REDIS_CONN
        )
    except Exception as e:
        logging.info(f"Error: {str(e)}")
        raise self.retry(countdown=10, exc=e, max_retries=3, args=(col, start_row, end_row))

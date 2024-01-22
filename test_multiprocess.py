import concurrent.futures
import math
import os
import numpy as np
import pickle
from datetime import datetime


def load_img_from_folder(folder_path):
    files = os.listdir(folder_path)
    load_img = np.zeros((6478, 5287, 122), dtype=np.float16)
    for file_name in files:
        file_path = os.path.join(folder_path, file_name)
        start_ix, end_ix = os.path.splitext(file_name)[0].split("-")
        start_ix = int(start_ix)
        end_ix = int(end_ix)
        load_img[start_ix: end_ix] = np.load(file_path)
    return load_img
    

def load_sklearn_model_to_file(file_path):
    with open(file_path, "rb") as file:
        return pickle.load(file)


def predict_N(data):
    clf_RF_1 = load_sklearn_model_to_file("./RF_save/clf_RF_1.pkl")
    row = data[0]
    col_and_bands = data[1]
    
    # num_cols = col_and_bands.shape[0]
    # N_data_in_row = np.zeros((num_cols))
    # for col in range(num_cols):
    #     bands = col_and_bands[col]
    #     format_bands = bands.astype(np.float32).reshape((1, bands.shape[0]))
    #     N_data_in_row[col] = clf_RF_1.predict(format_bands)
    # return row, N_data_in_row
        
    def my_func(bands):
        format_bands = bands.astype(np.float32).reshape((1, bands.shape[0]))
        return clf_RF_1.predict(format_bands)
    data = np.apply_along_axis(my_func, 1, col_and_bands)
    return row, data.reshape((data.shape[0]))

def main():
    img = load_img_from_folder("./img_save")
    
    num_rows = img.shape[0]
    num_cols = img.shape[1]
    N_data_in_pixel = np.zeros((num_rows, num_cols), np.float32)
    
    print(f"Finished load img data from folder")
    
    data_to_process = zip(range(num_rows), img)
    # print(data_to_process[0][0])
    # print(data_to_process[0][1].shape)
    print("finished prepare data")
    time_start = datetime.now()
    with concurrent.futures.ProcessPoolExecutor(8) as executor:
        for prime in executor.map(predict_N, data_to_process):
            row = prime[0]
            N_data = prime[1]
            print(f"Received data from row {row}")
            if row % 100 == 0:
                time_now = datetime.now()
                print(f"finidied calculate to row {row} in: {time_now - time_start}")
            
            N_data_in_pixel[row] = N_data
            
    print(N_data_in_pixel.shape)
    print(N_data_in_pixel[0])

if __name__ == '__main__':
    main()
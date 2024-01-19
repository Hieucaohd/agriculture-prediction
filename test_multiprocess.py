import concurrent.futures
import math
import os
import numpy as np
import pickle


def load_img_from_folder(folder_path):
    files = os.listdir(folder_path)
    load_img = np.zeros((6478, 5287, 122), dtype=np.float16)
    for file_name in files:
        file_path = f"{folder_path}/{file_name}"
        start_ix, end_ix = os.path.splitext(file_name)[0].split("-")
        start_ix = int(start_ix)
        end_ix = int(end_ix)
        load_img[start_ix: end_ix] = np.load(file_path)
    return load_img
    

def load_sklearn_model_to_file(file_path):
    with open(file_path, "rb") as file:
        return pickle.load(file)


def predict_N(col_and_bands):
    clf_RF_1 = load_sklearn_model_to_file("./RF_save/clf_RF_1.pkl")

    for bands in col_and_bands:
        return bands.shape

def main():
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for prime in executor.map(predict_N, img_small):
            print(prime)

if __name__ == '__main__':
    main()
import concurrent.futures
import math
import os
import numpy as np
import pickle


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
    row = data[0]
    col_and_bands = data[1]
    
    def my_func(bands):
        format_bands = bands.astype(np.float32).reshape((1, bands.shape[0]))
        return clf_RF_1.predict(format_bands)
    data = np.apply_along_axis(my_func, 1, col_and_bands)
    return row, data.reshape((data.shape[0]))

def main():
    img = load_img_from_folder("./img_save")
    clf_RF_1 = load_sklearn_model_to_file("./RF_save/clf_RF_1.pkl")
    print(f"Finished load img data from folder")
    
    num_rows = img.shape[0]
    num_cols = img.shape[1]

    def my_func(bands):
        format_bands = bands.astype(np.float32).reshape((1, bands.shape[0]))
        return clf_RF_1.predict(format_bands)
    
    result = np.apply_along_axis(my_func, 2, img)
    return result
        
    


if __name__ == '__main__':
    # def my_func(a):
    #     print(a)
    #     print(a.shape)
    
    # b = np.array([
    #     [[1, 1, 1],[2, 2, 2],[3, 3, 3]], 
    #     [[4, 4, 4],[5, 5, 5],[6, 6, 6]], 
    #     [[7, 7, 7],[8, 8, 8],[9, 9, 9]]
    # ])
    # print(np.apply_along_axis(my_func, 2, b))
    
    result = main()
    print(result.shape)

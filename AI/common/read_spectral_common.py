
# import các hàm cần thiết

import spectral.io.envi as envi		# hàm để đọc file .hdr và file .img
import numpy as np 					# để thao tác với ma trận
from spectral import open_image, imshow 	# để hiển thị hình ảnh
import pandas as pd # để đọc file excel
try:
    from typing import LiteralString
except:
    from typing_extensions import LiteralString
from sklearn.metrics import mean_squared_error #để tính Mean squared error
from sklearn import tree
import os
from sklearn.ensemble import RandomForestRegressor
from typing import Literal
import torch.nn as nn
import torch
from sklearn.feature_selection import mutual_info_regression
import dill
import warnings
warnings.filterwarnings("ignore")


def load_sklearn_model_to_file_by_cloudpickle(file_path):
    with open(file_path, 'rb') as f:
        return dill.load(f)


def get_full_path(path):
    script_dir = os.path.dirname(os.path.realpath(__file__))
    return os.path.abspath(os.path.join(script_dir, path))


class WrapSKlearnModelWithBand():
    def __init__(self, sklearn_model, bands_ix) -> None:
        self.sklearn_model = sklearn_model
        self.bands_ix = bands_ix
    
    def fit(self, X_train, Y_train):
        self.sklearn_model = self.sklearn_model.fit(X_train, Y_train)
    
    def predict(self, X_target):
        return self.sklearn_model.predict(X_target)


# mở file .hdr và file .img
img = envi.open(
    get_full_path("../../data/spectral_image/hyper_20220913_3cm.hdr"), 
    get_full_path("../../data/spectral_image/hyper_20220913_3cm.img")
)


def get_bands(row_num, col_num):
    return img[row_num, col_num]


# hàm này trả về các pixel nằm trong hình vuông có tọa độ (row_num, col_num) là tâm
# và độ dài cạnh là (2 * scopes + 1)
def pixel_in_scope(row_num, col_num, scopes):
    pixel_in_scope = set()

    for scope in range(1, scopes + 1):
        
        for i in range(row_num - scope, row_num + scope + 1):
            pixel_in_scope.add((i, col_num + scope))
            pixel_in_scope.add((i, col_num - scope))
            
        for i in range(col_num - scope, col_num + scope + 1):
            pixel_in_scope.add((row_num + scope, i))
            pixel_in_scope.add((row_num - scope, i))
            
    pixel_in_scope.add((row_num, col_num))
    return pixel_in_scope


# thông tin của map, tạm thời được đưa vào bằng tay
# "x1_pixel": tọa độ pixel x của điểm 1
# "y1_pixel": tọa độ pixel y của điểm 1
# "x1_axes": tọa độ trái đất x của điểm 1
# "y1_axes": tọa độ trái đất y của điểm 1
# "x_resolution": độ phân giải x của map
# "y_resolution": độ phân giải y của map
map_infor = {
    "col_1": 1.000,
    "row_1": 1.000,
    "east_1": 530499.467,
    "north_1": 2355871.685,
    "col_resolution": 3.0000000000e-002,
    "row_resolution": 3.0000000000e-002
}


# công thức lấy tọa độ trái đất khi biết tạo độ trái đất của điểm có tạo độ pixel (1, 1)
# (East - 530499.467 ) / 0.03 = số cột
# (2355871.685 - north) / 0.03 = cố hàng
# suy ra:
# East = 530499.467 + so cột * 0.03
# North = 2355871.685 - so hàng * 0.03 

# từ tọa độ pixel, lấy tọa độ trái đất
def get_axes(row_num, col_num):
    east = map_infor["east_1"] + col_num * map_infor["col_resolution"]  
    north = map_infor["north_1"] - row_num * map_infor["row_resolution"]
    return (north, east)
    

# từ tọa độ trái đất, lấy tọa độ pixel
def get_pixel(north, east):
    col_num = (east - map_infor["east_1"]) / map_infor["col_resolution"]
    row_num = (map_infor["north_1"] - north) / map_infor["row_resolution"]
    
    col_num = int(round(col_num, 0))
    row_num = int(round(row_num, 0))
    return (row_num, col_num)


# đọc file excel chứa data của NPK và tọa độ Trái Đất
data_df = pd.read_excel(
    get_full_path("../../data/spectral_image/DATA_Mua2_PhuTho_2022_3.xlsx")
)


# Lấy cột dữ liệu đo được của ngày 13/9/2022
data_df_13_09_2022 = data_df.loc[1:, "Unnamed: 14":"Chlorophyll-a.1"]


data_df_13_09_2022.rename(columns={
    "Unnamed: 14": "type_of_field"
}, inplace=True)


# Bỏ đi những dòng bị thiếu tên điểm 
data_df_13_09_2022 = data_df_13_09_2022[~pd.isna(data_df_13_09_2022["type_of_field"])]


# Tạo thêm cột 2 tọa độ pixel
data_df_13_09_2022["row_num"] = [None] * len(data_df_13_09_2022)
data_df_13_09_2022["col_num"] = [None] * len(data_df_13_09_2022)


# Định nghĩa hàm tính tọa độ pixel của điểm thực đo ngoài thực nghiệm
def fill_row_num_and_col_num(index, row, df: pd.DataFrame):
    east = row["East.1"]
    north = row["North.1"]
    row_num, col_num = get_pixel(north, east)
    df.at[index, "row_num"] = row_num
    df.at[index, "col_num"] = col_num


# Lặp hàm tính tọa độ pixel trên cho từng dòng
for index, row in data_df_13_09_2022.iterrows():
    fill_row_num_and_col_num(index, row, data_df_13_09_2022)


# Lấy giá trị max của từng band trong các ô xung quanh
def get_max_bands(row_num, col_num, scope):
    number_points = 2 * scope + 1
    band_in_scope = np.zeros((number_points ** 2, 122))
    for i, (row_num, col_num) in enumerate(pixel_in_scope(row_num, col_num, scope)):
        band_in_scope[i] = img[row_num, col_num, :].reshape(122)
    return np.max(band_in_scope, axis=0) # Lấy giá trị max của từng cột (band)


# Lấy giá trị max của từng band trong các ô xung quanh
def get_average_bands(row_num, col_num, scope):
    number_points = 2 * scope + 1
    band_in_scope = np.zeros((number_points ** 2, 122))
    for i, (row_num, col_num) in enumerate(pixel_in_scope(row_num, col_num, scope)):
        band_in_scope[i] = img[row_num, col_num, :].reshape(122)
    return np.average(band_in_scope, axis=0) # Lấy giá trị max của từng cột (band)


# Lấy giá trị max của từng band trong các ô xung quanh
def get_min_bands(row_num, col_num, scope):
    number_points = 2 * scope + 1
    band_in_scope = np.zeros((number_points ** 2, 122))
    for i, (row_num, col_num) in enumerate(pixel_in_scope(row_num, col_num, scope)):
        band_in_scope[i] = img[row_num, col_num, :].reshape(122)
    return np.min(band_in_scope, axis=0) # Lấy giá trị max của từng cột (band)


def generate_sample(df: pd.DataFrame, bands_ix: list[int], type_of_output: Literal["N", "P", "K"] = None, type_of_field: Literal["T", "J", "BC"] = None, func_to_cal_band = get_average_bands):
    if type_of_output == "N":
        output_column = "N conc. (mg/kg).1"
    elif type_of_output == "K":
        output_column = "K conc. (mg/kg).1"
    elif type_of_output == "P":
        output_column = "P conc. (mg/kg).1"
    
    sample = df[["type_of_field",output_column, "row_num", "col_num"]]
   
    for band_ix in bands_ix:
        sample[f"band_{band_ix}"] = [None] * len(sample)
    
    band_start_name = "band_" + str(bands_ix[0])
    band_end_name = "band_" + str(bands_ix[-1])


    # Thêm 122 giá trị max của 122 band vào tập data sample N
    for index, row in sample.iterrows():
        row_num = row["row_num"]
        col_num = row["col_num"]
        sample.loc[index, band_start_name:band_end_name] = func_to_cal_band(row_num, col_num, 3)[bands_ix]
    
    if type_of_field:
        if type(type_of_field) == list:
            total_condition = sample["type_of_field"].str.startswith(type_of_field[0])
            for field in type_of_field[1:]:
                condition = sample["type_of_field"].str.startswith(field)
                total_condition |= condition
            sample = sample[total_condition]
        else:
            sample = sample[sample["type_of_field"].str.startswith(type_of_field)] 
    sample.rename(columns={
        output_column: "target"
    }, inplace=True)
    return sample
    

def create_X_train_Y_train(df: pd.DataFrame, bands_ix: list[int]):
    band_start_name = "band_" + str(bands_ix[0])
    band_end_name = "band_" + str(bands_ix[-1])
    X = df.loc[:, band_start_name:band_end_name].to_numpy()
    Y = df.loc[:, "target"].to_numpy()

    X = X.astype(np.float32)
    Y = Y.astype(np.float32)

    total_sample = len(X)
    max_train = int(total_sample * 1)
    # max_val = int(total_sample * 1)

    X_train = torch.tensor(X[0:max_train])
    # X_val = torch.tensor(X[max_train:max_val])
    # X_test = torch.tensor(X[max_val:])

    Y_train = torch.tensor(Y[0:max_train])
    # Y_val = torch.tensor(Y[max_train:max_val])
    # Y_test = torch.tensor(Y[max_val:])

    Y_train = Y_train.reshape((Y_train.shape[0], 1))
    # Y_val = Y_val.reshape((Y_val.shape[0], 1))
    # Y_test = Y_test.reshape((Y_test.shape[0], 1))

    return X_train, Y_train


class NeutralNetwork(nn.Module):
    def __init__(self, number_bands: int, bands_ix, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.bands_ix = bands_ix
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(number_bands, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 1)
        )
    
    def forward(self, x):
        N_value = self.linear_relu_stack(x)
        return N_value


def train_model(model, loss_fn, optimizer, X_train, Y_train, X_val, Y_val, n_epochs, min_loss=0.5):
    for epoch in range(n_epochs):
        model.train()

        y_pred = model(X_train)
        loss = loss_fn(y_pred, Y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            model.eval()
            with torch.inference_mode():
                loss = np.sqrt(loss)
                # y_pred_val = model(X_val)
                # loss_val = np.sqrt(loss_fn(y_pred_val, Y_val))
                # print(f"Epoch: {epoch} | loss train: {loss} | loss val: {loss_val}")
                print(f"Epoch: {epoch} | loss train: {loss}")
                if loss < min_loss:
                    return


def calculate_mutual_info_for_all(df: pd.DataFrame, type_of_output: Literal["N", "P", "K"] = None, type_of_field: Literal["T", "J", "BC"] = None, func_to_cal_band = get_average_bands):
    bands_ix = list(range(0, 122))
    sample = generate_sample(df, bands_ix, type_of_output, type_of_field, func_to_cal_band)
    X_train, Y_train = create_X_train_Y_train(sample, bands_ix)
    mutual_infos = mutual_info_regression(X_train, Y_train, random_state=42)

    band_ix_and_mutual = []
    for band_ix, mutual_info in zip(bands_ix, mutual_infos):
        band_ix_and_mutual.append((band_ix, mutual_info))
    return sorted(band_ix_and_mutual, key=lambda data: data[1], reverse=True)


def get_bands_ix_from_mutual_info(df: pd.DataFrame, min_mutual_info: float, type_of_output: Literal["N", "P", "K"] = None, type_of_field: Literal["T", "J", "BC"] = None, func_to_cal_band = get_average_bands):
    band_ix_and_mutual = calculate_mutual_info_for_all(df, type_of_output, type_of_field, func_to_cal_band)
    band_ix_and_mutual_after = filter(lambda data:  data[1] > min_mutual_info, band_ix_and_mutual)
    bands_ix = [data[0] for data in band_ix_and_mutual_after]
    return sorted(bands_ix)


def predict_using_neutral_network(X_train, Y_train, X_target, Y_target, bands_ix, name_file_output, super_param={"lr": 0.0001, "weight_decay": 1e-5, "n_epochs": 40000, "stop_value": 0.5}, re_run="N"):
    loss_fn = nn.MSELoss()
    if not os.path.exists(name_file_output) or re_run == "Y":
        number_bands = X_train.shape[1]
        model = NeutralNetwork(number_bands, bands_ix)
        optimizer = torch.optim.Adam(model.parameters(), lr=super_param["lr"], weight_decay=super_param["weight_decay"])
        n_epochs = 40000
        train_model(model, loss_fn, optimizer, X_train, Y_train, [], [], n_epochs, 0.5)
    else:
        model = load_sklearn_model_to_file_by_cloudpickle(name_file_output)
    
    model.eval()
    with torch.inference_mode():
        loss_fn = nn.MSELoss()
        Y_target_pred = model(X_target)
        loss = loss_fn(Y_target, Y_target_pred)
        return np.sqrt(loss), Y_target_pred, model


def predict_using_random_forest(X_train, Y_train, X_target, Y_target, bands_ix, super_param={}):
    clf = RandomForestRegressor()
    RF_band = WrapSKlearnModelWithBand(clf, bands_ix)
    
    Y_train = Y_train.reshape(Y_train.shape[0])
    
    RF_band.fit(X_train, Y_train)
    Y_pred = RF_band.predict(X_target)
    loss = np.sqrt(mean_squared_error(Y_target, Y_pred))
    return loss, Y_pred, RF_band


def predict_using_decision_tree(X_train, Y_train, X_target, Y_target, bands_ix, super_param={}):
    clf = tree.DecisionTreeRegressor()
    DT_band = WrapSKlearnModelWithBand(clf, bands_ix)
    
    Y_train = Y_train.reshape(Y_train.shape[0])
    
    DT_band.fit(X_train, Y_train)
    Y_pred = DT_band.predict(X_target)
    loss = np.sqrt(mean_squared_error(Y_target, Y_pred))
    return loss, Y_pred, DT_band


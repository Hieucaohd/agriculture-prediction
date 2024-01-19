# %%
# import các hàm cần thiết

import spectral.io.envi as envi		# hàm để đọc file .hdr và file .img
import numpy as np 					# để thao tác với ma trận
from spectral import open_image, imshow 	# để hiển thị hình ảnh
import pandas as pd                         # để đọc file excel
from typing import LiteralString
from sklearn.metrics import mean_squared_error #để tính Mean squared error
from sklearn import tree
import os
from sklearn.ensemble import RandomForestRegressor
from typing import Literal
import torch.nn as nn
import torch
import warnings
warnings.filterwarnings("ignore")


# %%
# mở file .hdr và file .img
img = envi.open("hyper_20220913_3cm.hdr", "hyper_20220913_3cm.img")

# %%
# xem các thông tin cơ bản của file
img

# %%
# xem kích cỡ của ma trận
img.shape

# %%
def get_bands(row_num, col_num):
    return img[row_num, col_num]

# %%
# hiển thị hình ảnh chụp được
# view = imshow(img)

# %%
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

# %%
# lấy thông tin về map
with open("hyper_20220913_3cm.hdr", "r") as file:
    list_lines = file.read().split("\n")
    list_lines = filter(lambda line: "map info" in line, list_lines)
    map_infor = list(list_lines)[0]


# %%
map_infor

# %%
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

# %%
map_infor

# %%
# công thức lấy tọa độ trái đất khi biết tạo độ trái đất của điểm có tạo độ pixel (1, 1)
# (East - 530499.467 ) / 0.03 = số cột
# (2355871.685 - north) / 0.03 = cố hàng
# suy ra:
# East = 530499.467 + so cột * 0.03
# North = 2355871.685 - so hàng * 0.03 

# %%
# từ tọa độ pixel, lấy tọa độ trái đất
def get_axes(row_num, col_num):
    east = map_infor["east_1"] + col_num * map_infor["col_resolution"]  
    north = map_infor["north_1"] - row_num * map_infor["row_resolution"]
    return (north, east)
    

# %%
# từ tọa độ trái đất, lấy tọa độ pixel
def get_pixel(north, east):
    col_num = (east - map_infor["east_1"]) / map_infor["col_resolution"]
    row_num = (map_infor["north_1"] - north) / map_infor["row_resolution"]
    
    col_num = int(round(col_num, 0))
    row_num = int(round(row_num, 0))
    return (row_num, col_num)

# %%
# đọc file excel chứa data của NPK và tọa độ Trái Đất
data_df = pd.read_excel("DATA_Mua2_PhuTho_2022_3.xlsx")

# %%
data_df

# %%
# data_df.columns

# %%
# data_df.info()

# %%
# Lấy cột dữ liệu đo được của ngày 13/9/2022
data_df_13_09_2022 = data_df.loc[1:, "Unnamed: 14":"Chlorophyll-a.1"]

# %%
data_df_13_09_2022.rename(columns={
    "Unnamed: 14": "type_of_field"
}, inplace=True)

# %%
# Bỏ đi những dòng bị thiếu tên điểm 
data_df_13_09_2022 = data_df_13_09_2022[~pd.isna(data_df_13_09_2022["type_of_field"])]

# %%
# data_df_13_09_2022

# %%
# data_df_13_09_2022.info()

# %%
# Tạo thêm cột 2 tọa độ pixel
data_df_13_09_2022["row_num"] = [None] * len(data_df_13_09_2022)
data_df_13_09_2022["col_num"] = [None] * len(data_df_13_09_2022)

# %%
# data_df_13_09_2022

# %%
# data_df_13_09_2022.columns

# %%
# Định nghĩa hàm tính tọa độ pixel của điểm thực đo ngoài thực nghiệm
def fill_row_num_and_col_num(index, row, df: pd.DataFrame):
    east = row["East.1"]
    north = row["North.1"]
    row_num, col_num = get_pixel(north, east)
    df.at[index, "row_num"] = row_num
    df.at[index, "col_num"] = col_num

# %%
# Lặp hàm tính tọa độ pixel trên cho từng dòng
for index, row in data_df_13_09_2022.iterrows():
    fill_row_num_and_col_num(index, row, data_df_13_09_2022)

# %%
# data_df_13_09_2022

# %%
# data_df_13_09_2022.info()

# %%
# Tính trung bình 122 band  
def get_average_bands(row_num, col_num, scope):
    number_points = 2 * scope + 1
    band_in_scope = np.zeros((number_points ** 2, 122))
    for i, (row_num, col_num) in enumerate(pixel_in_scope(row_num, col_num, scope)):
        band_in_scope[i] = img[row_num, col_num, :].reshape(122)
    return np.average(band_in_scope, axis=0)

# %%
def generate_sample(df: pd.DataFrame, type_of_output: Literal["N", "P", "K"] = None, type_of_field: Literal["T", "J", "BC"] = None):
    if type_of_output == "N":
        output_column = "N conc. (mg/kg).1"
    elif type_of_output == "K":
        output_column = "K conc. (mg/kg).1"
    elif type_of_output == "P":
        output_column = "P conc. (mg/kg).1"
    
    sample = df[["type_of_field",output_column, "row_num", "col_num"]]

    #Tạo bảng gồm giá trị của N, tọa độ pixel điểm thực nghiệm và 122 giá trị trống
    for i in range(122):
        sample[f"band_{i}"] = [None] * len(sample)

    # Thêm 122 giá trị trung bình của 122 band vào tập data sample N
    for index, row in sample.iterrows():
        row_num = row["row_num"]
        col_num = row["col_num"]
        sample.loc[index, "band_0":"band_121"] = get_average_bands(row_num, col_num, 3) 
    
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
    

# %%
# dùng deep learning

# %%
def create_X_train_Y_train(df: pd.DataFrame):
    X = df.loc[:, "band_0":"band_121"].to_numpy()
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

# %%
class NeutralNetwork(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(122, 100),
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

# %%
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
                # # print(f"Epoch: {epoch} | loss train: {loss} | loss val: {loss_val}")
                # print(f"Epoch: {epoch} | loss train: {loss}")
                if loss < min_loss:
                    return

# %%
RERUN_NN_MODEL = "N"

# %%
def predict_using_neutral_network(X_train, Y_train, X_target, Y_target, name_file_output, super_param={"lr": 0.0001, "weight_decay": 1e-5, "n_epochs": 40000, "stop_value": 0.5}, re_run="N"):
    loss_fn = nn.MSELoss()
    if not os.path.exists(name_file_output) or re_run == "Y" or RERUN_NN_MODEL == "Y":
        model = NeutralNetwork()
        optimizer = torch.optim.Adam(model.parameters(), lr=super_param["lr"], weight_decay=super_param["weight_decay"])
        n_epochs = 40000
        train_model(model, loss_fn, optimizer, X_train, Y_train, [], [], n_epochs, 0.5)
        torch.jit.script(model).save(name_file_output)
    else:
        model = torch.jit.load(name_file_output)
    
    model.eval()
    with torch.inference_mode():
        loss_fn = nn.MSELoss()
        Y_target_pred = model(X_target)
        loss = loss_fn(Y_target, Y_target_pred)
        return np.sqrt(loss), Y_target_pred, model

# %%
def predict_using_random_forest(X_train, Y_train, X_target, Y_target, super_param={}):
    clf = RandomForestRegressor()
    Y_train = Y_train.reshape(Y_train.shape[0])
    clf = clf.fit(X_train, Y_train)
    Y_pred = clf.predict(X_target)
    loss = np.sqrt(mean_squared_error(Y_target, Y_pred))
    return loss, Y_pred, clf
    

# %%
def predict_using_decision_tree(X_train, Y_train, X_target, Y_target, super_param={}):
    clf = tree.DecisionTreeRegressor()
    Y_train = Y_train.reshape(Y_train.shape[0])
    clf = clf.fit(X_train, Y_train)
    Y_pred = clf.predict(X_target)
    loss = np.sqrt(mean_squared_error(Y_target, Y_pred))
    return loss, Y_pred, clf
    

# %%
#Dự đoán Nito

# %%
target_value = "N"
train_field = "T"
sample = generate_sample(data_df_13_09_2022, target_value, train_field)
X_train, Y_train = create_X_train_Y_train(sample)
sample_target = generate_sample(data_df_13_09_2022, target_value, "BC")
X_target, Y_target = create_X_train_Y_train(sample_target)
super_param={"lr": 0.01, "weight_decay": 1e-5, "n_epochs": 40000, "stop_value": 0.5}
re_run = "N"
loss_NN, pred_NN, model_NN_1 = predict_using_neutral_network(X_train, Y_train, X_target, Y_target, f"{'_'.join(list(train_field))}_model_predict_{target_value}.pt", super_param, re_run)
# print(f"{loss_NN=}")
# print(f"{pred_NN=}")
loss_RF, pred_RF, clf_RF_1 = predict_using_random_forest(X_train, Y_train, X_target, Y_target, super_param)
# print(f"{loss_RF=}")
# print(f"{pred_RF=}")
loss_DT, pred_DT, clf_DT_1 = predict_using_decision_tree(X_train, Y_train, X_target, Y_target, super_param)
# print(f"{loss_DT=}")
# print(f"{pred_DT=}")

# %%
target_value = "N"
train_field = "J"
sample = generate_sample(data_df_13_09_2022, target_value, train_field)
X_train, Y_train = create_X_train_Y_train(sample)
sample_target = generate_sample(data_df_13_09_2022, target_value, "BC")
X_target, Y_target = create_X_train_Y_train(sample_target)
super_param={"lr": 0.0001, "weight_decay": 1e-5, "n_epochs": 40000, "stop_value": 0.5}
re_run = "N"
loss_NN, pred_NN, model_NN_2 = predict_using_neutral_network(X_train, Y_train, X_target, Y_target, f"{'_'.join(list(train_field))}_model_predict_{target_value}.pt", super_param, re_run)
# print(f"{loss_NN=}")
# print(f"{pred_NN=}")
loss_RF, pred_RF, clf_RF_2 = predict_using_random_forest(X_train, Y_train, X_target, Y_target, super_param)
# print(f"{loss_RF=}")
# print(f"{pred_RF=}")
loss_DT, pred_DT, clf_DT_2 = predict_using_decision_tree(X_train, Y_train, X_target, Y_target, super_param)
# print(f"{loss_DT=}")
# print(f"{pred_DT=}")

# %%
target_value = "N"
train_field = ["T", "J"]
sample = generate_sample(data_df_13_09_2022, target_value, train_field)
X_train, Y_train = create_X_train_Y_train(sample)
sample_target = generate_sample(data_df_13_09_2022, target_value, "BC")
X_target, Y_target = create_X_train_Y_train(sample_target)
super_param={"lr": 0.0001, "weight_decay": 1e-5, "n_epochs": 40000, "stop_value": 0.5}
re_run = "N"
loss_NN, pred_NN, model_NN_3 = predict_using_neutral_network(X_train, Y_train, X_target, Y_target, f"{'_'.join(list(train_field))}_model_predict_{target_value}.pt", super_param, re_run)
# print(f"{loss_NN=}")
# print(f"{pred_NN=}")
loss_RF, pred_RF, clf_RF_3 = predict_using_random_forest(X_train, Y_train, X_target, Y_target, super_param)
# print(f"{loss_RF=}")
# print(f"{pred_RF=}")
loss_DT, pred_DT, clf_DT_3 = predict_using_decision_tree(X_train, Y_train, X_target, Y_target, super_param)
# print(f"{loss_DT=}")
# print(f"{pred_DT=}")

# %%
# Dự đoán Photpho

# %%
target_value = "P"
train_field = "T"
sample = generate_sample(data_df_13_09_2022, target_value, train_field)
X_train, Y_train = create_X_train_Y_train(sample)
sample_target = generate_sample(data_df_13_09_2022, target_value, "BC")
X_target, Y_target = create_X_train_Y_train(sample_target)
super_param={"lr": 0.0001, "weight_decay": 1e-5, "n_epochs": 40000, "stop_value": 0.5}
re_run = "N"
loss_NN, pred_NN, model_NN_4 = predict_using_neutral_network(X_train, Y_train, X_target, Y_target, f"{'_'.join(list(train_field))}_model_predict_{target_value}.pt", super_param, re_run)
# print(f"{loss_NN=}")
# print(f"{pred_NN=}")
loss_RF, pred_RF, clf_RF_4 = predict_using_random_forest(X_train, Y_train, X_target, Y_target, super_param)
# print(f"{loss_RF=}")
# print(f"{pred_RF=}")
loss_DT, pred_DT, clf_DT_4 = predict_using_decision_tree(X_train, Y_train, X_target, Y_target, super_param)
# print(f"{loss_DT=}")
# print(f"{pred_DT=}")

# %%
target_value = "P"
train_field = "J"
sample = generate_sample(data_df_13_09_2022, target_value, train_field)
X_train, Y_train = create_X_train_Y_train(sample)
sample_target = generate_sample(data_df_13_09_2022, target_value, "BC")
X_target, Y_target = create_X_train_Y_train(sample_target)
super_param={"lr": 0.0001, "weight_decay": 1e-5, "n_epochs": 40000, "stop_value": 0.5}
re_run = "N"
loss_NN, pred_NN, model_NN_5 = predict_using_neutral_network(X_train, Y_train, X_target, Y_target, f"{'_'.join(list(train_field))}_model_predict_{target_value}.pt", super_param, re_run)
# print(f"{loss_NN=}")
# print(f"{pred_NN=}")
loss_RF, pred_RF, clf_RF_5 = predict_using_random_forest(X_train, Y_train, X_target, Y_target, super_param)
# print(f"{loss_RF=}")
# print(f"{pred_RF=}")
loss_DT, pred_DT, clf_DT_5 = predict_using_decision_tree(X_train, Y_train, X_target, Y_target, super_param)
# print(f"{loss_DT=}")
# print(f"{pred_DT=}")

# %%
target_value = "P"
train_field = ["T", "J"]
sample = generate_sample(data_df_13_09_2022, target_value, train_field)
X_train, Y_train = create_X_train_Y_train(sample)
sample_target = generate_sample(data_df_13_09_2022, target_value, "BC")
X_target, Y_target = create_X_train_Y_train(sample_target)
super_param={"lr": 0.0001, "weight_decay": 1e-5, "n_epochs": 40000, "stop_value": 0.5}
re_run = "N"
loss_NN, pred_NN, model_NN_6 = predict_using_neutral_network(X_train, Y_train, X_target, Y_target, f"{'_'.join(list(train_field))}_model_predict_{target_value}.pt", super_param, re_run)
# print(f"{loss_NN=}")
# print(f"{pred_NN=}")
loss_RF, pred_RF, clf_RF_6 = predict_using_random_forest(X_train, Y_train, X_target, Y_target, super_param)
# print(f"{loss_RF=}")
# print(f"{pred_RF=}")
loss_DT, pred_DT, clf_DT_6 = predict_using_decision_tree(X_train, Y_train, X_target, Y_target, super_param)
# print(f"{loss_DT=}")
# print(f"{pred_DT=}")

# %%
#Dự đoán Kali

# %%
target_value = "K"
train_field = "T"
sample = generate_sample(data_df_13_09_2022, target_value, train_field)
X_train, Y_train = create_X_train_Y_train(sample)
sample_target = generate_sample(data_df_13_09_2022, target_value, "BC")
X_target, Y_target = create_X_train_Y_train(sample_target)
super_param={"lr": 0.0001, "weight_decay": 1e-5, "n_epochs": 40000, "stop_value": 0.5}
re_run = "N"
loss_NN, pred_NN, model_NN_7 = predict_using_neutral_network(X_train, Y_train, X_target, Y_target, f"{'_'.join(list(train_field))}_model_predict_{target_value}.pt", super_param, re_run)
# print(f"{loss_NN=}")
# print(f"{pred_NN=}")
loss_RF, pred_RF, clf_RF_7 = predict_using_random_forest(X_train, Y_train, X_target, Y_target, super_param)
# print(f"{loss_RF=}")
# print(f"{pred_RF=}")
loss_DT, pred_DT, clf_DT_7 = predict_using_decision_tree(X_train, Y_train, X_target, Y_target, super_param)
# print(f"{loss_DT=}")
# print(f"{pred_DT=}")

# %%
target_value = "K"
train_field = "J"
sample = generate_sample(data_df_13_09_2022, target_value, train_field)
X_train, Y_train = create_X_train_Y_train(sample)
sample_target = generate_sample(data_df_13_09_2022, target_value, "BC")
X_target, Y_target = create_X_train_Y_train(sample_target)
super_param={"lr": 0.0001, "weight_decay": 1e-5, "n_epochs": 40000, "stop_value": 0.5}
re_run = "N"
loss_NN, pred_NN, model_NN_8 = predict_using_neutral_network(X_train, Y_train, X_target, Y_target, f"{'_'.join(list(train_field))}_model_predict_{target_value}.pt", super_param, re_run)
# print(f"{loss_NN=}")
# print(f"{pred_NN=}")
loss_RF, pred_RF, clf_RF_8 = predict_using_random_forest(X_train, Y_train, X_target, Y_target, super_param)
# print(f"{loss_RF=}")
# print(f"{pred_RF=}")
loss_DT, pred_DT, clf_DT_8 = predict_using_decision_tree(X_train, Y_train, X_target, Y_target, super_param)
# print(f"{loss_DT=}")
# print(f"{pred_DT=}")

# %%
target_value = "K"
train_field = ["T", "J"]
sample = generate_sample(data_df_13_09_2022, target_value, train_field)
X_train, Y_train = create_X_train_Y_train(sample)
sample_target = generate_sample(data_df_13_09_2022, target_value, "BC")
X_target, Y_target = create_X_train_Y_train(sample_target)
super_param={"lr": 0.0001, "weight_decay": 1e-5, "n_epochs": 40000, "stop_value": 0.5}
re_run = "N"
loss_NN, pred_NN, model_NN_9 = predict_using_neutral_network(X_train, Y_train, X_target, Y_target, f"{'_'.join(list(train_field))}_model_predict_{target_value}.pt", super_param, re_run)
# print(f"{loss_NN=}")
# print(f"{pred_NN=}")
loss_RF, pred_RF, clf_RF_9 = predict_using_random_forest(X_train, Y_train, X_target, Y_target, super_param)
# print(f"{loss_RF=}")
# print(f"{pred_RF=}")
loss_DT, pred_DT, clf_DT_9 = predict_using_decision_tree(X_train, Y_train, X_target, Y_target, super_param)
# print(f"{loss_DT=}")
# print(f"{pred_DT=}")

# %%
X_train.shape

# %%
img[1235, 2345].shape

# %%
img.shape

# %%
np.zeros((4, 4, 5))

# %%
np.zeros((4, 4))

# %%
import numpy as np

# Your matrix
matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Define your function
def your_function(x):
    # Apply your logic here
    return x ** 2

# Vectorize the function
vectorized_function = np.vectorize(your_function)

# Apply the vectorized function to each element in the matrix
result_matrix = vectorized_function(matrix)

# print(result_matrix)


# %%
import functools

# %%
img.shape

# %%
def predict_all(model, img):
    num_row = img.shape[0]
    num_col = img.shape[1]
    result_matrix = np.zeros((num_row, num_col))
    for row in range(num_row):
        for col in range(num_col):
            bands = img[row, col]
            format_bands = bands.astype(np.float32).reshape((1, bands.shape[0]))
            result_matrix[row, col] = model.predict(format_bands)
    return result_matrix
    

# %%
img_small = img[:4,:4,:]

# %%
img_small.shape

# %%
# result = predict_all(clf_RF_1, img_small)

# %%
def calculate_N(col_and_bands):
    # print(col_and_bands.shape)
    return col_and_bands.shape

# %%
# import numpy as np
# import matplotlib.pyplot as plt

# # Assuming 'assigned_values' is your array with assigned values for each pixel
# # Replace this with your actual data
# assigned_values = np.random.randint(0, 255, size=(100, 100))

# # Define a colormap (you can choose any colormap from matplotlib)
# cmap = 'viridis'

# # Create a figure and axis
# fig, ax = plt.subplots()

# # Display the image with the assigned colors
# im = ax.imshow(result, cmap=cmap)

# # Add a colorbar to the right of the plot
# cbar = fig.colorbar(im, ax=ax)

# # Show the plot
# plt.show()




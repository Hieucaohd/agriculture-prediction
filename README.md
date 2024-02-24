# Agriculture Prediction

Agriculture prediction is a project that use AI to predict Nito, Photpho, Kali from spectral image.

<https://www.spectralpython.net/>

[![GitHub Actions Build](https://github.com/apache/spark/actions/workflows/build_main.yml/badge.svg)](https://github.com/apache/spark/actions/workflows/build_main.yml)
[![PySpark Coverage](https://codecov.io/gh/apache/spark/branch/master/graph/badge.svg)](https://codecov.io/gh/apache/spark)
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/pyspark?period=month&units=international_system&left_color=black&right_color=orange&left_text=PyPI%20downloads)](https://pypi.org/project/pyspark/)


## Install package dependencies

To install the package dependencies, in linux run:

```bash
sudo chmod +x ./src/bin/*.sh
./src/bin/install.sh
```

For windown:

```bash
./src/bin/window_install.ps1
```


## How to train AI model

We use PyTorch and Scikit-learn library to train the model inference: 
1) PyTorch: <https://pytorch.org/>

2) Scikit-learn: <https://scikit-learn.org/stable/>

Here is module that we build to train our AI: [train AI model.](https://github.com/Hieucaohd/agriculture-prediction/blob/main/AI/common/read_spectral_common.py)



## How to calculate image

1) Read spectral image then save each column data to file: [convert spectral image to numpy array.](https://github.com/Hieucaohd/agriculture-prediction/blob/main/src/convert_img_to_np.ipynb)

2) Load AI model then predict Nito, Photpho, Kali for each column by that model: [load AI model to code.](https://github.com/Hieucaohd/agriculture-prediction/blob/main/src/bulk_calculate.ipynb)

3) Draw image: [draw image.](https://github.com/Hieucaohd/agriculture-prediction/blob/main/src/draw_img.ipynb)



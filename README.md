# Agriculture Prediction

Agriculture prediction is a project that use AI to predict Nito, Photpho, Kali from spectral image.

What is data structured of spectral image? <https://www.spectralpython.net/>


## Install package dependencies

To install the package dependencies and create folder that source code used, in linux run:

```bash
sudo chmod +x ./src/bin/*.sh
./src/bin/install.sh
```

For windown:

```bash
./src/bin/window_install.ps1
```

Then download spectral images in these links and save to folder ./data/spectral_image
1) Train data: [train file.](https://docs.google.com/spreadsheets/d/10Wp1fz59lR28xio-lvEcxIZ8OE09b267/edit?usp=sharing&ouid=101687776546423364812&rtpof=true&sd=true)
2) Header image file: [header file.](https://drive.google.com/file/d/1-FeYM1thYKsi6yO2wcq_kHSVfwjpz9ki/view?usp=sharing)
3) Bands image file: [bands file.]()

## How to read spectral image
1) Open terminal in Linux or power-shell in Window.
2) Make sure that you are in environment of project.

	In Linux:
	```bash
	source venv/bin/activate
	```
	
	In Window:
	```bash
	.\venv\bin\activate
	```
4) Type this command in terminal:

	```bash
	ipython --pylab
	```
4) After you in interactive mode of ipython, run this code:
	```python
	import spectral.io.envi as envi
	img = envi.open("./data/spectral_image/hyper_20220913_3cm.hdr", "./data/spectral_image/hyper_20220913_3cm.img")
	from spectral import open_image, imshow
	view = imshow(img)
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



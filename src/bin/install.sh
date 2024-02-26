#!/bin/bash

cd "$(dirname $0)/../../"

mkdir -p data/spectral_image
mkdir -p data/spectral_image_1
mkdir -p data/spectral_image_2
mkdir -p RF_save
mkdir -p src/checkpoint
mkdir -p src/log/celery
mkdir -p src/run/celery
mkdir -p src/proj/db/db_new
mkdir -p src/data/img_col_data
mkdir -p src/data/saved_result
mkdir -p src/data/img_result_saved
mkdir -p model_saved/NN_save/using_mutual_information

source venv/bin/activate

pip install -r requirements.txt

pip install -U Celery==5.3.6

deactivate

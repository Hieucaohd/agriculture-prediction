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

source venv/bin/activate

pip install -r requirements.txt

pip install -U Celery==5.3.6

deactivate

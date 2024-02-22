#!/bin/bash


source /home/ubuntu/code/agriculture-prediction/venv/bin/activate

pip install -r /home/ubuntu/code/agriculture-prediction/requirements.txt

pip install -U Celery==5.3.6

deactivate

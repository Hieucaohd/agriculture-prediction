#!/bin/bash

source /home/ubuntu/code/agriculture-prediction/venv/bin/activate
celery -A proj events


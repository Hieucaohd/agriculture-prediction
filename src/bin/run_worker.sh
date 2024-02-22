#!/bin/bash

rm /home/ubuntu/code/agriculture-prediction/src/log/celery/*.log
source /home/ubuntu/code/agriculture-prediction/venv/bin/activate
celery -A proj worker --concurrency=$1 -E -l INFO \
	--pidfile=/home/ubuntu/code/agriculture-prediction/src/run/celery/%n.pid \
	--logfile=/home/ubuntu/code/agriculture-prediction/src/log/celery/%n%I.log

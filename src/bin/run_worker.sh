#!/bin/bash

cd "$(dirname $0)/../"

rm log/celery/*.log
source ../venv/bin/activate
celery -A proj worker --concurrency=$1 -E -l INFO \
	--pidfile=run/celery/%n.pid \
	--logfile=log/celery/%n%I.log

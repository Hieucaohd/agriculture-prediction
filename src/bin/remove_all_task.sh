#!/bin/bash

cd "$(dirname $0)/../"

source ../venv/bin/activate
celery -A proj purge $1

#!/bin/bash

cd "$(dirname $0)/../"

tail -f log/celery/*.log

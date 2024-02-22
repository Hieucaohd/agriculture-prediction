from celery import Celery
from celery import Task

app = Celery(
	'proj',
	broker='redis://localhost:6379/0',
	# backend='db+postgresql://celery:celery@localhost:5432/celery_backend',
	# backend='rpc://',
	include=['proj.tasks']
)

app.conf.update(
	result_expires=2
)

if __name__ == "__main__":
    app.start()

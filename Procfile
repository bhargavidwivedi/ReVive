web: gunicorn core.wsgi --settings=core.settings
worker: celery -A core worker --loglevel=info --pool=solo
beat: celery -A core beat --loglevel=info
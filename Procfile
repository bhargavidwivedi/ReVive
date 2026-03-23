web: DJANGO_SETTINGS_MODULE=core.settings gunicorn core.wsgi
worker: DJANGO_SETTINGS_MODULE=core.settings celery -A core worker --loglevel=info --pool=solo
beat: DJANGO_SETTINGS_MODULE=core.settings celery -A core beat --loglevel=info
#!/bin/bash
exec gunicorn core.wsgi:application --bind 0.0.0.0:${PORT:-8080} --workers 2
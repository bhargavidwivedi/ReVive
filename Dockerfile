FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install whitenoise gunicorn

COPY . .

RUN python manage.py collectstatic --noinput --settings=core.settings

RUN chmod +x start.sh

EXPOSE 8080

CMD ["/bin/bash", "start.sh"]
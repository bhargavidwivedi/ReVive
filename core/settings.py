from dotenv import load_dotenv
load_dotenv()
from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent.parent

SECRET_KEY   = os.getenv("SECRET_KEY", "django-insecure-revive-hospital-readmission-2024")
DEBUG        = os.getenv("DEBUG", "True") == "True"
ALLOWED_HOSTS= os.getenv("ALLOWED_HOSTS", "*").split(",")

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "rest_framework",
    "predictor",
    "corsheaders",
]

MIDDLEWARE = [
    "corsheaders.middleware.CorsMiddleware",
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
]

ROOT_URLCONF = "core.urls"
TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [BASE_DIR / "templates"],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": BASE_DIR / "db.sqlite3",
    }
}

STATIC_URL = "/static/"
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"
CORS_ALLOW_ALL_ORIGINS = True
REDIS_URL             = os.getenv("REDIS_URL", "redis://localhost:6379/0")
CELERY_BROKER_URL     = REDIS_URL
CELERY_RESULT_BACKEND = REDIS_URL
CELERY_ACCEPT_CONTENT     = ["json"]
CELERY_TASK_SERIALIZER    = "json"
CELERY_RESULT_SERIALIZER  = "json"
CELERY_TIMEZONE           = "Asia/Kolkata"

INSTALLED_APPS += [
    "django_celery_beat",
    "django_celery_results",
]

CELERY_BEAT_SCHEDULE = {
    "score-all-patients-daily": {
        "task"    : "predictor.tasks.score_all_patients",
        "schedule": 86400,
    },
}
EMAIL_BACKEND       = "django.core.mail.backends.smtp.EmailBackend"
EMAIL_HOST          = "smtp.gmail.com"
EMAIL_PORT          = 587
EMAIL_USE_TLS       = True
EMAIL_HOST_USER     = "bhargavidwivedi56@gmail.com"
EMAIL_HOST_PASSWORD = "kuekwfdbypiuonkd"
DEFAULT_FROM_EMAIL  = "bhargavidwivedi56@gmail.com"
CELERY_BEAT_SCHEDULER = "django_celery_beat.schedulers:DatabaseScheduler"
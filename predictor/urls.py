from django.urls import path
from . import views

urlpatterns = [
    path("predict/",               views.predict,                  name="predict"),
    path("predict/batch/",         views.predict_batch,            name="predict_batch"),
    path("health/",                views.health,                   name="health"),
    path("discharge/",             views.trigger_discharge_scoring,name="discharge"),
    path("test-celery/",           views.test_celery_view,         name="test-celery"),
    path("fhir/search/",           views.fhir_search,              name="fhir-search"),
    path("fhir/<str:patient_id>/", views.fhir_predict,             name="fhir-predict"),
    path("analyze-notes/",         views.analyze_notes,            name="analyze-notes"),
    path("care-pathway/",          views.care_pathway,             name="care-pathway"),
    path("outcomes/",      views.record_patient_outcome, name="outcomes"),
    path("drift/",         views.drift_detection,         name="drift"),
    path("retrain/",       views.trigger_retraining,      name="retrain"),
    path("system-health/", views.system_health,           name="system-health"),
]
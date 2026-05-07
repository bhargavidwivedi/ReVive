from django.db import models

class PatientOutcome(models.Model):
    patient_id          = models.CharField(max_length=100)
    predicted_risk      = models.FloatField()
    actually_readmitted = models.BooleanField(default=False)
    risk_level          = models.CharField(max_length=20, default="Medium")
    notes               = models.TextField(blank=True)
    recorded_at         = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-recorded_at"]

    def __str__(self):
        return f"Patient {self.patient_id} — {self.risk_level}"

class PredictionLog(models.Model):
    session_id          = models.CharField(max_length=100, blank=True)
    readmission_prob    = models.FloatField()
    risk_level          = models.CharField(max_length=20)
    age                 = models.IntegerField(default=0)
    los                 = models.IntegerField(default=0)
    predicted_at        = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-predicted_at"]

    def __str__(self):
        return f"{self.risk_level} ({self.readmission_prob:.2%}) at {self.predicted_at}"
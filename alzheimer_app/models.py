from django.db import models
from django.utils import timezone


class PatientAssessment(models.Model):
    GENDER_CHOICES = [('M', 'Male'), ('F', 'Female'), ('O', 'Other')]
    APOE4_CHOICES = [(0, '0 alleles'), (1, '1 allele'), (2, '2 alleles')]
    RESULT_CHOICES = [
        ('healthy', 'Cognitively Normal'),
        ('mci', 'Mild Cognitive Impairment'),
        ('alzheimer', "Alzheimer's Disease"),
    ]

    patient_name = models.CharField(max_length=100, blank=True, default='Anonymous')
    patient_age = models.IntegerField()
    gender = models.CharField(max_length=1, choices=GENDER_CHOICES)
    created_at = models.DateTimeField(default=timezone.now)

    mmse_score = models.FloatField()
    cdr_score = models.FloatField()
    education_years = models.FloatField()
    socioeconomic_status = models.IntegerField()
    etiv = models.FloatField()
    nwbv = models.FloatField()
    asf = models.FloatField()

    memory_score = models.FloatField()
    attention_score = models.FloatField()
    language_score = models.FloatField()
    visuospatial_score = models.FloatField()

    family_history = models.BooleanField(default=False)
    apoe4_alleles = models.IntegerField(choices=APOE4_CHOICES, default=0)

    prediction = models.CharField(max_length=20, choices=RESULT_CHOICES, blank=True)
    confidence_healthy = models.FloatField(default=0.0)
    confidence_mci = models.FloatField(default=0.0)
    confidence_alzheimer = models.FloatField(default=0.0)
    risk_score = models.FloatField(default=0.0)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.patient_name} | {self.prediction} | {self.created_at.strftime('%Y-%m-%d')}"

    @property
    def result_label(self):
        labels = {'healthy': 'Cognitively Normal', 'mci': 'Mild Cognitive Impairment', 'alzheimer': "Alzheimer's Disease"}
        return labels.get(self.prediction, 'Unknown')

    @property
    def result_color(self):
        return {'healthy': '#10b981', 'mci': '#f59e0b', 'alzheimer': '#ef4444'}.get(self.prediction, '#6b7280')

    @property
    def risk_level(self):
        if self.risk_score < 33: return 'Low'
        elif self.risk_score < 66: return 'Moderate'
        return 'High'

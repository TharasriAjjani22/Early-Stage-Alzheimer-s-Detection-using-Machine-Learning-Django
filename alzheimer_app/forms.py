from django import forms
from .models import PatientAssessment


class AssessmentForm(forms.ModelForm):
    class Meta:
        model = PatientAssessment
        fields = [
            'patient_name', 'patient_age', 'gender',
            'mmse_score', 'cdr_score', 'education_years', 'socioeconomic_status',
            'etiv', 'nwbv', 'asf',
            'memory_score', 'attention_score', 'language_score', 'visuospatial_score',
            'family_history', 'apoe4_alleles',
        ]
        widgets = {
            'patient_name': forms.TextInput(attrs={'placeholder': 'Enter patient name', 'class': 'form-input'}),
            'patient_age': forms.NumberInput(attrs={'min': 50, 'max': 100, 'class': 'form-input'}),
            'gender': forms.Select(attrs={'class': 'form-input'}),
            'mmse_score': forms.NumberInput(attrs={'min': 0, 'max': 30, 'step': '0.5', 'class': 'form-input'}),
            'cdr_score': forms.Select(choices=[(0, '0 – No Impairment'), (0.5, '0.5 – Questionable'), (1, '1 – Mild'), (2, '2 – Moderate'), (3, '3 – Severe')], attrs={'class': 'form-input'}),
            'education_years': forms.NumberInput(attrs={'min': 0, 'max': 25, 'step': '1', 'class': 'form-input'}),
            'socioeconomic_status': forms.Select(choices=[(1,'1 – Low'),(2,'2 – Lower-Middle'),(3,'3 – Middle'),(4,'4 – Upper-Middle'),(5,'5 – High')], attrs={'class': 'form-input'}),
            'etiv': forms.NumberInput(attrs={'min': 900, 'max': 2200, 'step': '10', 'class': 'form-input', 'placeholder': '900–2200 cm³'}),
            'nwbv': forms.NumberInput(attrs={'min': 0.60, 'max': 0.95, 'step': '0.001', 'class': 'form-input', 'placeholder': '0.60–0.95'}),
            'asf': forms.NumberInput(attrs={'min': 0.8, 'max': 1.8, 'step': '0.01', 'class': 'form-input', 'placeholder': '0.80–1.80'}),
            'memory_score': forms.NumberInput(attrs={'min': 0, 'max': 100, 'class': 'form-input'}),
            'attention_score': forms.NumberInput(attrs={'min': 0, 'max': 100, 'class': 'form-input'}),
            'language_score': forms.NumberInput(attrs={'min': 0, 'max': 100, 'class': 'form-input'}),
            'visuospatial_score': forms.NumberInput(attrs={'min': 0, 'max': 100, 'class': 'form-input'}),
            'family_history': forms.CheckboxInput(attrs={'class': 'form-checkbox'}),
            'apoe4_alleles': forms.Select(attrs={'class': 'form-input'}),
        }
        labels = {
            'mmse_score': 'MMSE Score (0–30)',
            'cdr_score': 'CDR Rating',
            'education_years': 'Years of Education',
            'socioeconomic_status': 'Socioeconomic Status',
            'etiv': 'Est. Intracranial Volume (eTIV)',
            'nwbv': 'Normalized Whole Brain Volume',
            'asf': 'Atlas Scaling Factor (ASF)',
            'memory_score': 'Memory Score',
            'attention_score': 'Attention Score',
            'language_score': 'Language Score',
            'visuospatial_score': 'Visuospatial Score',
            'family_history': 'Family History of Alzheimer\'s',
            'apoe4_alleles': 'APOE-ε4 Allele Count',
        }

    def clean_mmse_score(self):
        val = self.cleaned_data.get('mmse_score')
        if val is not None and not (0 <= val <= 30):
            raise forms.ValidationError("MMSE must be between 0 and 30.")
        return val

    def clean_nwbv(self):
        val = self.cleaned_data.get('nwbv')
        if val is not None and not (0.5 <= val <= 1.0):
            raise forms.ValidationError("NWBV must be between 0.50 and 1.00.")
        return val

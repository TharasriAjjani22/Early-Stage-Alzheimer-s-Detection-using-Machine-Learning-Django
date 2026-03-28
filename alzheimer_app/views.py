from django.shortcuts import render, redirect, get_object_or_404
from django.http import JsonResponse
from django.views.decorators.http import require_POST
from django.db.models import Count
from .forms import AssessmentForm
from .models import PatientAssessment
from .ml_service import predict, get_feature_importances
import json


def dashboard(request):
    total = PatientAssessment.objects.count()
    by_result = PatientAssessment.objects.values('prediction').annotate(count=Count('id'))
    counts = {r['prediction']: r['count'] for r in by_result}
    recent = PatientAssessment.objects.all()[:8]
    importances = get_feature_importances()[:8]
    feature_labels = [f[0].replace('_', ' ').title() for f in importances]
    feature_values = [round(f[1] * 100, 1) for f in importances]
    context = {
        'total': total,
        'healthy_count': counts.get('healthy', 0),
        'mci_count': counts.get('mci', 0),
        'alzheimer_count': counts.get('alzheimer', 0),
        'recent_assessments': recent,
        'feature_labels': json.dumps(feature_labels),
        'feature_values': json.dumps(feature_values),
    }
    return render(request, 'alzheimer_app/dashboard.html', context)


def assessment(request):
    form = AssessmentForm()
    return render(request, 'alzheimer_app/assessment.html', {'form': form})


@require_POST
def run_assessment(request):
    form = AssessmentForm(request.POST)
    if form.is_valid():
        instance = form.save(commit=False)
        result = predict(request.POST)
        instance.prediction = result['prediction']
        instance.confidence_healthy = result['confidence_healthy']
        instance.confidence_mci = result['confidence_mci']
        instance.confidence_alzheimer = result['confidence_alzheimer']
        instance.risk_score = result['risk_score']
        instance.save()
        return redirect('result', pk=instance.pk)
    return render(request, 'alzheimer_app/assessment.html', {'form': form})


def result(request, pk):
    a = get_object_or_404(PatientAssessment, pk=pk)
    recommendations = get_recommendations(a.prediction, a.risk_score)
    return render(request, 'alzheimer_app/result.html', {'assessment': a, 'recommendations': recommendations})


def history(request):
    assessments = PatientAssessment.objects.all()
    return render(request, 'alzheimer_app/history.html', {'assessments': assessments})


def api_predict(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        result = predict(data)
        return JsonResponse({'success': True, **result})
    return JsonResponse({'error': 'POST required'}, status=405)


def get_recommendations(prediction, risk_score):
    base = [
        "Schedule regular follow-up appointments with a neurologist.",
        "Maintain a socially active lifestyle and engage in cognitive exercises.",
        "Ensure regular cardiovascular exercise (150 min/week).",
        "Follow a Mediterranean-style diet rich in omega-3 fatty acids.",
        "Prioritize 7-8 hours of quality sleep per night.",
    ]
    if prediction == 'mci':
        return [
            "Mild Cognitive Impairment detected. Immediate specialist referral recommended.",
            "Begin structured cognitive training programs (memory games, puzzles).",
            "Monitor for progression — repeat assessment every 6 months.",
            "Discuss pharmacological options with your physician.",
        ] + base[:3]
    elif prediction == 'alzheimer':
        return [
            "Early Alzheimer's indicators detected. Urgent specialist evaluation required.",
            "Comprehensive neuropsychological battery testing is recommended.",
            "Discuss acetylcholinesterase inhibitor therapy with your neurologist.",
            "Establish care planning and caregiver support resources.",
            "Consider brain MRI/PET scan for structural confirmation.",
        ]
    return [
        "No significant cognitive impairment detected.",
        "Continue preventive lifestyle measures to maintain brain health.",
    ] + base

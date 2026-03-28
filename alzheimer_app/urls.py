from django.urls import path
from . import views

urlpatterns = [
    path('', views.dashboard, name='dashboard'),
    path('assessment/', views.assessment, name='assessment'),
    path('assessment/run/', views.run_assessment, name='run_assessment'),
    path('result/<int:pk>/', views.result, name='result'),
    path('history/', views.history, name='history'),
    path('api/predict/', views.api_predict, name='api_predict'),
]

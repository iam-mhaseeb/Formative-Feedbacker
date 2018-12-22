from django.urls import path, include, re_path
from feedbacker import views

app_name = 'dashboard'

urlpatterns = [
   path('', views.index, name='landing'),
   path('feedback', views.feedback, name='feedback'),
]
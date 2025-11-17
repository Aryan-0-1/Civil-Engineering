from django.urls import path
from . import views

urlpatterns = [
    path('', views.water_quality, name='water_quality'),
    path('/water_predict',views.water_predict,name='water_predict'),
]
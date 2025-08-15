from django.contrib import admin
from django.urls import path,include
from . import views

urlpatterns = [
    path('',views.page,name='page'),
    path('pipe_calculation/',views.pipe_calculation,name='pipe_calculation'),
]
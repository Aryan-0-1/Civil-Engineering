from django.contrib import admin
from django.urls import path,include
from . import views

urlpatterns = [
    path('',views.oc,name='oc'),
    path('calculate/',views.calculate,name='calculate'),
]


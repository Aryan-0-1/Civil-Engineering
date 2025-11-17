from django.contrib import admin
from django.urls import path,include
from . import views

urlpatterns = [
    path('',views.page,name='page'),
    # path('de/',views.pipe_calculation,name='pipe_calculation'),
    path('design_view/',views.design_view,name='design_view'),
]
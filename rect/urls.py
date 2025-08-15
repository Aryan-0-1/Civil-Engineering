from django.urls import path
from . import views

urlpatterns = [
    path('', views.experiment, name='experiment'),
    # path('results/', views.results, name='results'),
    path('reset/', views.reset, name='reset'),
    path('static/download-manual/', views.download_manual, name='download_manual'),
    path('download-results/', views.download_results, name='download_results'),
    # path('manual/', views.manual_view, name='manual'),
    # path('download-pdf/', views.download_pdf, name='download_pdf'),  # New URL
    path('manual.pdf', views.serve_pdf, name='serve_pdf'),  # New URL for PDF

]
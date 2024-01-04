from django.urls import path
from . import views1
from . import views2
from . import views3

urlpatterns = [
    # ... other patterns
    path('upload-pdf/', views1.upload_pdf_view, name='upload-pdf'),
    path('get-pdf/', views2.query_pdf_view, name='get-pdf'),
    path('upload-audio/', views3.upload_audio, name='upload-audio'),
    # ... other patterns
]

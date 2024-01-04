# myapp/models.py

from django.db import models

class ProcessedPDF(models.Model):
    file = models.FileField(upload_to='processed_pdfs/')
    result_text = models.TextField()

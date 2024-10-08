# Generated by Django 5.0 on 2023-12-07 06:15

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("myapp", "0001_initial"),
    ]

    operations = [
        migrations.CreateModel(
            name="ProcessedPDF",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("file", models.FileField(upload_to="processed_pdfs/")),
                ("result_text", models.TextField()),
            ],
        ),
        migrations.DeleteModel(
            name="UploadedPDF",
        ),
    ]

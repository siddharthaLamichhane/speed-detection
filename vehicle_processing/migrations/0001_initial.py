# Generated by Django 5.1.7 on 2025-03-08 03:06

import django.core.validators
import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='User',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('username', models.CharField(max_length=100)),
                ('email', models.EmailField(max_length=254, unique=True)),
                ('license_plate', models.CharField(max_length=20, unique=True, validators=[django.core.validators.RegexValidator(message='Enter a valid license plate.', regex='^[A-Za-z0-9]{1,20}$')])),
            ],
        ),
        migrations.CreateModel(
            name='Violation',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('vehicle_type', models.CharField(max_length=50)),
                ('speed', models.FloatField()),
                ('timestamp', models.DateTimeField(auto_now_add=True)),
                ('license_plate', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='vehicle_processing.user')),
            ],
            options={
                'indexes': [models.Index(fields=['license_plate'], name='vehicle_pro_license_b1e94b_idx'), models.Index(fields=['timestamp'], name='vehicle_pro_timesta_976279_idx')],
            },
        ),
    ]

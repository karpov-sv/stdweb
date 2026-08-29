from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('stdweb', '0016_task_groups'),
    ]

    operations = [
        migrations.AddField(
            model_name='task',
            name='celery_steps',
            field=models.JSONField(blank=True, default=list),
        ),
    ]

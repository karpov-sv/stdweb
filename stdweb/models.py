from django.db import models
from django.db.models.signals import pre_delete
from django.dispatch import receiver

import os, shutil

from . import settings

class Task(models.Model):
    # path = models.CharField(max_length=250, blank=False, unique=True, editable=False) # Base dir where task processing will be performed
    original_name = models.CharField(max_length=250, blank=False) # Original filename
    title = models.CharField(max_length=250, blank=True) # Optional title or comment

    state = models.CharField(max_length=50, blank=False, default='initial') # State of the task

    created = models.DateTimeField(auto_now_add=True)
    modified = models.DateTimeField(auto_now=True)

    config = models.JSONField(default=dict) #

    def path(self):
        return os.path.join(settings.TASKS_PATH, str(self.id))


@receiver(pre_delete, sender=Task)
def delete_task_hook(sender, instance, using, **kwargs):
    path = instance.path()

    # Cleanup the data on filesystem related to this model
    if os.path.exists(path):
        shutil.rmtree(path)
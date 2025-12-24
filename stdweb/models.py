from django.db import models
from django.db.models.signals import pre_delete
from django.dispatch import receiver
from django.utils.timezone import now
from django.contrib.auth.models import User
from django.conf import settings

import os, shutil
import datetime


class Task(models.Model):
    # path = models.CharField(max_length=250, blank=False, unique=True, editable=False) # Base dir where task processing will be performed
    original_name = models.CharField(max_length=250, blank=False) # Original filename
    title = models.CharField(max_length=250, blank=True) # Optional title or comment

    state = models.CharField(max_length=50, blank=False, default='initial') # State of the task

    celery_id = models.CharField(max_length=50, blank=True, null=True, default=None, editable=False) # Celery task ID, when running
    celery_chain_ids = models.JSONField(default=list, blank=True) # List of all subtask IDs in chain
    celery_pid = models.IntegerField(blank=True, null=True, default=None, editable=False) # PID of the Celery worker process

    user =  models.ForeignKey(User, on_delete=models.CASCADE)

    created = models.DateTimeField(auto_now_add=True)
    modified = models.DateTimeField(auto_now=True) # Updated on every .save()
    completed = models.DateTimeField(default=now, editable=False) # Manually updated on finishing the processing

    config = models.JSONField(default=dict, blank=True) #

    def path(self):
        return os.path.join(settings.TASKS_PATH, str(self.id))

    def complete(self):
        self.completed = now()

    def __str__(self):
        return f"{self.id}: {self.user.username} : {self.original_name}"

    class Meta:
        permissions = [
            ('skyportal_upload', 'Can upload the task results to SkyPortal'),
            ('view_all_tasks', 'Can view tasks from all users'),
        ]


@receiver(pre_delete, sender=Task)
def delete_task_hook(sender, instance, using, **kwargs):
    path = instance.path()

    # Cleanup the data on filesystem related to this model
    if os.path.exists(path):
        shutil.rmtree(path)


class Preset(models.Model):
    name = models.CharField(max_length=250, blank=False) # Preset name
    config = models.JSONField(default=dict, blank=True, help_text='Initial config for the task, in JSON format')
    files = models.TextField(blank=True, help_text='Files to be copied into new task, one per line') # Files to be copied into new task, one per line

    def __str__(self):
        return f"{self.id}: {self.name}"

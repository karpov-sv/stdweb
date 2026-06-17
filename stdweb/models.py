from django.db import models
from django.db.models.signals import pre_delete
from django.dispatch import receiver
from django.utils.timezone import now
from django.contrib.auth.models import User, Group
from django.conf import settings

import os, shutil
import datetime


def accessible_groups(user):
    """Groups a user may share tasks with: their own groups, or all for staff."""
    if not user.is_authenticated:
        return Group.objects.none()
    if user.is_staff:
        return Group.objects.all()
    return user.groups.all()


class Task(models.Model):
    # path = models.CharField(max_length=250, blank=False, unique=True, editable=False) # Base dir where task processing will be performed
    original_name = models.CharField(max_length=250, blank=False) # Original filename
    title = models.CharField(max_length=250, blank=True) # Optional title or comment

    state = models.CharField(max_length=50, blank=False, default='initial') # State of the task

    celery_id = models.CharField(max_length=50, blank=True, null=True, default=None, editable=False) # Celery task ID, when running
    celery_chain_ids = models.JSONField(default=list, blank=True) # List of all subtask IDs in chain
    celery_pid = models.IntegerField(blank=True, null=True, default=None, editable=False) # PID of the Celery worker process

    user =  models.ForeignKey(User, on_delete=models.CASCADE)

    # Groups this task is shared with. Every member of a listed group may
    # view and edit the task, in addition to the owner. See can_view()/can_edit().
    groups = models.ManyToManyField(Group, blank=True, related_name='tasks')

    created = models.DateTimeField(auto_now_add=True)
    modified = models.DateTimeField(auto_now=True) # Updated on every .save()
    completed = models.DateTimeField(default=now, editable=False) # Manually updated on finishing the processing

    # For positional searches
    ra = models.FloatField(blank=True, null=True)
    dec = models.FloatField(blank=True, null=True)
    radius = models.FloatField(blank=True, null=True)
    moc = models.TextField(blank=True, null=True)

    config = models.JSONField(default=dict, blank=True) #

    def path(self):
        return os.path.join(settings.TASKS_PATH, str(self.id))

    def complete(self):
        self.completed = now()

    # --- Access control -----------------------------------------------------
    # All task access checks funnel through these helpers so that the rules
    # live in exactly one place. The matching queryset filter is
    # Task.accessible_to() below.

    def is_shared_with(self, user):
        """Whether `user` belongs to any group this task is shared with."""
        return self.groups.filter(user=user).exists()

    def can_view(self, user):
        """Whether `user` may read this task."""
        if not user.is_authenticated:
            return False
        if user.is_staff or user == self.user:
            return True
        if user.has_perm('stdweb.view_all_tasks') or user.has_perm('stdweb.edit_all_tasks'):
            return True
        return self.is_shared_with(user)

    def can_edit(self, user):
        """Whether `user` may modify / submit / run this task."""
        if not user.is_authenticated:
            return False
        if user.is_staff or user == self.user:
            return True
        if user.has_perm('stdweb.edit_all_tasks'):
            return True
        return self.is_shared_with(user)

    def can_delete(self, user):
        """Whether `user` may delete this task. Only the owner or staff."""
        if not user.is_authenticated:
            return False
        return user.is_staff or user == self.user

    @staticmethod
    def accessible_to(user, queryset=None):
        """Queryset of tasks `user` may view. Mirrors can_view() at the DB level."""
        if queryset is None:
            queryset = Task.objects.all()
        if not user.is_authenticated:
            return queryset.none()
        if user.is_staff or user.has_perm('stdweb.view_all_tasks') or user.has_perm('stdweb.edit_all_tasks'):
            return queryset
        # Own tasks plus those shared with any of the user's groups.
        return queryset.filter(models.Q(user=user) | models.Q(groups__user=user)).distinct()

    def __str__(self):
        return f"{self.id}: {self.user.username} : {self.original_name}"

    class Meta:
        permissions = [
            ('skyportal_upload', 'Can upload the task results to SkyPortal'),
            ('view_all_tasks', 'Can view tasks from all users'),
            ('edit_all_tasks', 'Can modify tasks from all users'),
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


class ActionLog(models.Model):
    """Model for logging user actions on tasks."""

    ACTION_TYPES = [
        ('task_create', 'Task Created'),
        ('task_delete', 'Task Deleted'),
        ('task_duplicate', 'Task Duplicated'),
        ('task_update', 'Task Updated'),
        ('task_archive', 'Task Archived'),
        ('task_cleanup', 'Task Cleanup'),
        ('processing_start', 'Processing Started'),
        ('processing_complete', 'Processing Completed'),
        ('processing_failed', 'Processing Failed'),
        ('processing_cancel', 'Processing Cancelled'),
        ('file_upload', 'File Uploaded'),
        ('config_change', 'Config Changed'),
    ]

    timestamp = models.DateTimeField(auto_now_add=True, db_index=True)
    user = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True)
    action = models.CharField(max_length=50, choices=ACTION_TYPES, db_index=True)
    task = models.ForeignKey(Task, on_delete=models.SET_NULL, null=True, blank=True)
    task_id_ref = models.IntegerField(null=True, blank=True, help_text='Preserved task ID after task deletion')
    details = models.JSONField(default=dict, blank=True)
    ip_address = models.GenericIPAddressField(null=True, blank=True)

    class Meta:
        ordering = ['-timestamp']
        indexes = [
            models.Index(fields=['user', 'timestamp']),
            models.Index(fields=['action', 'timestamp']),
        ]

    def __str__(self):
        return f"{self.timestamp:%Y-%m-%d %H:%M:%S} - {self.user} - {self.get_action_display()}"

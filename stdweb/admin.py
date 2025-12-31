from django.contrib import admin
from django import forms
from django.utils.safestring import mark_safe
import json

from .models import Task
from .models import Preset
from .models import ActionLog

from .forms import PrettyJSONEncoder

class TaskAdmin(admin.ModelAdmin):
    search_fields = ['original_name', 'title', 'user__username']
    list_display = ['id', 'user', 'state', 'original_name', 'title']
    list_display_links = list_display

    def get_form(self, request, obj=None, **kwargs):
        form = super().get_form(request, obj, **kwargs)
        form.base_fields['config'].encoder = PrettyJSONEncoder

        return form


admin.site.register(Task, TaskAdmin)


class PresetAdmin(admin.ModelAdmin):
    list_display = ['id', 'name', 'config']
    list_display_links = list_display

    def get_form(self, request, obj=None, **kwargs):
        form = super().get_form(request, obj, **kwargs)
        form.base_fields['config'].encoder = PrettyJSONEncoder

        return form


admin.site.register(Preset, PresetAdmin)


@admin.register(ActionLog)
class ActionLogAdmin(admin.ModelAdmin):
    list_display = ['admin_timestamp', 'user', 'action', 'task_id_ref']
    list_filter = ['action', 'user', 'timestamp']
    search_fields = ['user__username', 'task_id_ref']
    readonly_fields = ['timestamp', 'user', 'action', 'task', 'task_id_ref', 'admin_details', 'ip_address']
    exclude = ['details']
    date_hierarchy = 'timestamp'

    def has_add_permission(self, request):
        return False  # Logs are created programmatically only

    def has_change_permission(self, request, obj=None):
        return False  # Logs are immutable

    def has_delete_permission(self, request, obj=None):
        return request.user.is_superuser  # Only superusers can delete logs

    @admin.display(description='Timestamp')
    def admin_timestamp(self, obj):
        return obj.timestamp.strftime('%Y-%m-%d %H:%M:%S')

    @admin.display(description='Details')
    def admin_details(self, obj):
        if obj.details:
            formatted = json.dumps(obj.details, indent=2, ensure_ascii=False)
            return mark_safe(f'<pre style="margin:0">{formatted}</pre>')
        return '-'

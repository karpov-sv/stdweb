from django.contrib import admin
from django import forms

from .models import Task
from .models import Preset

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

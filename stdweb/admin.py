from django.contrib import admin
from django.contrib.auth.models import User, Group
from django.contrib.auth.admin import UserAdmin, GroupAdmin
from django.contrib.admin.widgets import FilteredSelectMultiple
from django.shortcuts import render
from django import forms
from django.utils.safestring import mark_safe
import json
import humanize

from .models import Task
from .models import Preset
from .models import ActionLog

from .forms import PrettyJSONEncoder

class TaskAdmin(admin.ModelAdmin):
    search_fields = ['original_name', 'title', 'user__username']
    list_display = ['id', 'user', 'state', 'original_name', 'title']
    list_display_links = list_display
    filter_horizontal = ['groups']

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
    readonly_fields = ['admin_timestamp', 'user', 'action', 'task', 'task_id_ref', 'admin_details', 'ip_address']
    exclude = ['timestamp', 'details']
    date_hierarchy = 'timestamp'

    def has_add_permission(self, request):
        return False  # Logs are created programmatically only

    def has_change_permission(self, request, obj=None):
        return False  # Logs are immutable

    def has_delete_permission(self, request, obj=None):
        return request.user.is_superuser  # Only superusers can delete logs

    @admin.display(description='Timestamp, UTC')
    def admin_timestamp(self, obj):
        return obj.timestamp.strftime('%Y-%m-%d %H:%M:%S - ' + humanize.naturaltime(obj.timestamp))

    @admin.display(description='Details')
    def admin_details(self, obj):
        if obj.details:
            formatted = json.dumps(obj.details, indent=2, ensure_ascii=False)
            return mark_safe(f'<pre style="margin:0">{formatted}</pre>')
        return '-'


# =============================================================================
# Bulk group membership management
# =============================================================================

class GroupActionForm(forms.Form):
    """Group picker shown on the intermediate page of the user-list actions."""
    group = forms.ModelChoiceField(queryset=Group.objects.all(), label="Group")


class CustomUserAdmin(UserAdmin):
    """User admin with bulk 'add/remove selected users to/from a group' actions."""

    actions = ['add_to_group', 'remove_from_group']

    @admin.action(description="Add selected users to group…")
    def add_to_group(self, request, queryset):
        return self._apply_group_action(request, queryset, add=True)

    @admin.action(description="Remove selected users from group…")
    def remove_from_group(self, request, queryset):
        return self._apply_group_action(request, queryset, add=False)

    def _apply_group_action(self, request, queryset, add):
        action = 'add_to_group' if add else 'remove_from_group'

        if 'apply' in request.POST:
            form = GroupActionForm(request.POST)
            if form.is_valid():
                group = form.cleaned_data['group']
                for user in queryset:
                    if add:
                        user.groups.add(group)
                    else:
                        user.groups.remove(group)
                verb = 'added to' if add else 'removed from'
                self.message_user(request,
                                  f"{queryset.count()} user(s) {verb} group '{group.name}'.")
                return None  # back to the changelist
        else:
            form = GroupActionForm()

        return render(request, 'admin/group_action.html', {
            **self.admin_site.each_context(request),
            'title': "Add users to group" if add else "Remove users from group",
            'users': queryset,
            'form': form,
            'action': action,
            'add': add,
            'opts': self.model._meta,
        })


class WrappedFilteredSelectMultiple(FilteredSelectMultiple):
    # Real filter_horizontal fields (e.g. permissions) are wrapped by the admin
    # in a RelatedFieldWidgetWrapper, so their <select> lives in its own
    # container <div>. SelectFilter2.js builds the dual-list UI by prepending it
    # into the <select>'s parent node; with that wrapper the field <label> is a
    # sibling one level up and stays ahead of the selector. A bare
    # FilteredSelectMultiple has no wrapper, so the <select> shares its parent
    # with the <label> and the JS-inserted selector ends up *before* it.
    # (Django 6.0's admin template masks this by rendering the label as a
    # <legend> when use_fieldset is set; 5.2's template ignores use_fieldset, so
    # the bug shows in production.) Wrapping the widget output in a container
    # <div> mirrors the permissions field and fixes it on every Django version.
    use_fieldset = True

    def render(self, name, value, attrs=None, renderer=None):
        html = super().render(name, value, attrs=attrs, renderer=renderer)
        return mark_safe(f'<div class="related-widget-wrapper">{html}</div>')


class GroupAdminForm(forms.ModelForm):
    """Group form with a dual-list selector for managing members directly."""
    users = forms.ModelMultipleChoiceField(
        queryset=User.objects.all(),
        required=False,
        widget=WrappedFilteredSelectMultiple('users', is_stacked=False),
    )

    class Meta:
        model = Group
        fields = ['name', 'permissions']

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.instance and self.instance.pk:
            self.fields['users'].initial = self.instance.user_set.all()

    def save(self, commit=True):
        group = super().save(commit=commit)
        if commit:
            group.user_set.set(self.cleaned_data['users'])
        else:
            old_save_m2m = self.save_m2m

            def save_m2m():
                old_save_m2m()
                group.user_set.set(self.cleaned_data['users'])

            self.save_m2m = save_m2m
        return group


class CustomGroupAdmin(GroupAdmin):
    form = GroupAdminForm


admin.site.unregister(User)
admin.site.unregister(Group)
admin.site.register(User, CustomUserAdmin)
admin.site.register(Group, CustomGroupAdmin)

"""
Tests for the group-sharing UI: the task_update_groups web endpoint and the
groups field on the upload form.
"""

import pytest
from django.urls import reverse
from django.contrib.auth.models import Group

from stdweb import forms


@pytest.mark.django_db
class TestTaskUpdateGroups:
    """Tests for the task_update_groups AJAX endpoint."""

    def test_owner_can_share(self, client, user, task, shared_group):
        # Sharing is scoped to the user's own groups
        user.groups.add(shared_group)
        client.force_login(user)
        resp = client.post(reverse('task_update_groups', kwargs={'id': task.id}),
                           {'groups': [shared_group.id]})
        assert resp.status_code == 200
        assert resp.json()['success'] is True
        assert list(task.groups.all()) == [shared_group]

    def test_owner_can_unshare(self, client, user, task, shared_group):
        user.groups.add(shared_group)
        task.groups.add(shared_group)
        client.force_login(user)
        resp = client.post(reverse('task_update_groups', kwargs={'id': task.id}), {})
        assert resp.status_code == 200
        assert task.groups.count() == 0

    def test_non_editor_denied(self, client, other_user, task, shared_group):
        client.force_login(other_user)
        resp = client.post(reverse('task_update_groups', kwargs={'id': task.id}),
                           {'groups': [shared_group.id]})
        assert resp.status_code == 403
        assert task.groups.count() == 0

    def test_running_task_blocked(self, client, user, task, shared_group):
        task.celery_id = 'running'
        task.save()
        client.force_login(user)
        resp = client.post(reverse('task_update_groups', kwargs={'id': task.id}),
                           {'groups': [shared_group.id]})
        assert resp.status_code == 400

    def test_member_cannot_assign_unrelated_group(self, client, group_member, task, shared_group):
        """An editor may only toggle groups they themselves belong to."""
        task.groups.add(shared_group)  # gives group_member edit access
        unrelated = Group.objects.create(name='unrelated')
        client.force_login(group_member)
        resp = client.post(reverse('task_update_groups', kwargs={'id': task.id}),
                           {'groups': [shared_group.id, unrelated.id]})
        assert resp.status_code == 200
        assert unrelated not in task.groups.all()
        assert shared_group in task.groups.all()

    def test_member_unshare_preserves_invisible_group(self, client, group_member, task, shared_group):
        """Clearing the checklist must not drop groups the editor cannot see."""
        invisible = Group.objects.create(name='invisible')
        task.groups.add(shared_group, invisible)  # member is only in shared_group
        client.force_login(group_member)
        resp = client.post(reverse('task_update_groups', kwargs={'id': task.id}), {})
        assert resp.status_code == 200
        assert shared_group not in task.groups.all()
        assert invisible in task.groups.all()


@pytest.mark.django_db
class TestTaskListGroupSearch:
    """The task list filter should match on group name."""

    def test_filter_by_group_name(self, client, user, task, shared_group):
        task.groups.add(shared_group)
        client.force_login(user)
        resp = client.get(reverse('tasks', kwargs={'id': None}), {'query': shared_group.name})
        assert task in list(resp.context['tasks'])

    def test_filter_by_other_name_excludes(self, client, user, task, shared_group):
        task.groups.add(shared_group)
        client.force_login(user)
        resp = client.get(reverse('tasks', kwargs={'id': None}), {'query': 'nomatch'})
        assert task not in list(resp.context['tasks'])


@pytest.mark.django_db
class TestBulkGroupActions:
    """Bulk add/remove of selected tasks to/from groups via tasks_actions."""

    def _post(self, client, action, task_ids, group_ids):
        return client.post(reverse('tasks_actions'), {
            'action': action,
            'tasks': task_ids,
            'action_groups': group_ids,
            'referer': '/tasks/',
        })

    def test_bulk_add(self, client, user, task, shared_group):
        user.groups.add(shared_group)
        client.force_login(user)
        resp = self._post(client, 'add_to_groups', [task.id], [shared_group.id])
        assert resp.status_code == 302
        assert shared_group in task.groups.all()

    def test_bulk_remove(self, client, user, task, shared_group):
        user.groups.add(shared_group)
        task.groups.add(shared_group)
        client.force_login(user)
        resp = self._post(client, 'remove_from_groups', [task.id], [shared_group.id])
        assert resp.status_code == 302
        assert shared_group not in task.groups.all()

    def test_bulk_add_ignores_inaccessible_group(self, client, user, task, shared_group):
        user.groups.add(shared_group)
        other = Group.objects.create(name='inaccessible')
        client.force_login(user)
        self._post(client, 'add_to_groups', [task.id], [shared_group.id, other.id])
        assert shared_group in task.groups.all()
        assert other not in task.groups.all()

    def test_bulk_add_skips_non_editable_task(self, client, group_member, other_user_task, shared_group):
        """A user may only re-share tasks they can edit."""
        client.force_login(group_member)
        self._post(client, 'add_to_groups', [other_user_task.id], [shared_group.id])
        assert shared_group not in other_user_task.groups.all()

    def test_bulk_no_groups_selected(self, client, user, task, shared_group):
        user.groups.add(shared_group)
        task.groups.add(shared_group)
        client.force_login(user)
        resp = self._post(client, 'add_to_groups', [task.id], [])
        assert resp.status_code == 302
        # Nothing changed
        assert shared_group in task.groups.all()


@pytest.mark.django_db
class TestUploadFormGroups:
    """Tests for the groups field on the upload form."""

    def test_field_scoped_to_user_groups(self, user, shared_group):
        other = Group.objects.create(name='other')
        user.groups.add(shared_group)
        form = forms.UploadFileForm(user=user)
        choices = set(form.fields['groups'].queryset)
        assert shared_group in choices
        assert other not in choices

    def test_staff_sees_all_groups(self, staff_user, shared_group):
        other = Group.objects.create(name='other')
        form = forms.UploadFileForm(user=staff_user)
        choices = set(form.fields['groups'].queryset)
        assert shared_group in choices and other in choices

    def test_field_dropped_when_no_groups(self, user):
        """Users with no shareable groups don't get the field at all."""
        form = forms.UploadFileForm(user=user)
        assert 'groups' not in form.fields

    def test_renders_as_dropdown_not_plain_select(self, user, shared_group):
        """The groups field renders as a Bootstrap dropdown of checkboxes."""
        user.groups.add(shared_group)
        form = forms.UploadFileForm(user=user)
        html = str(form['groups'])
        assert 'dropdown-menu' in html
        assert 'type="checkbox"' in html
        assert shared_group.name in html
        assert '<select' not in html

"""
Tests for bulk group-membership management in the Django admin:
the user-list add/remove actions and the group-page members selector.
"""

import pytest
from django.urls import reverse
from django.contrib.auth.models import User

from stdweb.admin import GroupAdminForm


@pytest.fixture
def superuser(db):
    return User.objects.create_superuser('admin', 'admin@example.com', 'pass12345')


@pytest.mark.django_db
class TestUserGroupActions:
    """The 'Add/Remove selected users to/from group' user-list actions."""

    def _changelist(self):
        return reverse('admin:auth_user_changelist')

    def test_intermediate_page_rendered(self, client, superuser, user):
        client.force_login(superuser)
        resp = client.post(self._changelist(), {
            'action': 'add_to_group',
            '_selected_action': [user.id],
        })
        assert resp.status_code == 200
        assert b'added to' in resp.content
        assert b'name="group"' in resp.content

    def test_add_users_to_group(self, client, superuser, user, other_user, shared_group):
        client.force_login(superuser)
        resp = client.post(self._changelist(), {
            'action': 'add_to_group',
            '_selected_action': [user.id, other_user.id],
            'group': shared_group.id,
            'apply': '1',
        })
        assert resp.status_code == 302
        assert shared_group in user.groups.all()
        assert shared_group in other_user.groups.all()

    def test_remove_users_from_group(self, client, superuser, user, other_user, shared_group):
        user.groups.add(shared_group)
        other_user.groups.add(shared_group)
        client.force_login(superuser)
        resp = client.post(self._changelist(), {
            'action': 'remove_from_group',
            '_selected_action': [user.id, other_user.id],
            'group': shared_group.id,
            'apply': '1',
        })
        assert resp.status_code == 302
        assert shared_group not in user.groups.all()
        assert shared_group not in other_user.groups.all()


@pytest.mark.django_db
class TestGroupAdminForm:
    """The members ('users') selector on the group edit page."""

    def test_initial_members_prefilled(self, user, shared_group):
        user.groups.add(shared_group)
        form = GroupAdminForm(instance=shared_group)
        assert user in form.fields['users'].initial

    def test_save_sets_members(self, user, other_user, shared_group):
        form = GroupAdminForm(
            data={'name': shared_group.name, 'users': [user.id]},
            instance=shared_group,
        )
        assert form.is_valid(), form.errors
        form.save()
        assert user in shared_group.user_set.all()
        assert other_user not in shared_group.user_set.all()

    def test_save_can_clear_members(self, user, shared_group):
        user.groups.add(shared_group)
        form = GroupAdminForm(
            data={'name': shared_group.name, 'users': []},
            instance=shared_group,
        )
        assert form.is_valid(), form.errors
        form.save()
        assert shared_group.user_set.count() == 0

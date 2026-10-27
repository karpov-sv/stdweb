"""
Tests for key=value criteria on task config keywords in the task list filter.
"""

import pytest
from django.urls import reverse

from stdweb import utils
from stdweb.views_tasks import JSONKeyType
from stdweb.models import Task


class TestParseKeywordQuery:
    def test_plain_text(self):
        assert utils.parse_keyword_query('foo bar') == ('foo bar', [])

    def test_mixed(self):
        text, criteria = utils.parse_keyword_query('foo filter=R gain>=1.5 bar n<3 flag=true')
        assert text == 'foo bar'
        assert criteria == [('filter', '=', 'R'), ('gain', '>=', 1.5), ('n', '<', 3), ('flag', '=', True)]

    def test_quoted(self):
        text, criteria = utils.parse_keyword_query('''title="some thing" time>'2024-01-01 10:00' x="30"''')
        assert text == ''
        assert criteria == [('title', '=', 'some thing'), ('time', '>', '2024-01-01 10:00'), ('x', '=', '30')]

    def test_not_keywords(self):
        # Keys must start with a letter, and coordinates are left intact
        text, criteria = utils.parse_keyword_query('1abc=2 =3 12:30:00 +10:00:00')
        assert criteria == []
        assert text == '1abc=2 =3 12:30:00 +10:00:00'


class TestJSONKeyType:
    """SQL generated for the backends not used in tests."""

    class Compiler:
        def compile(self, expr):
            return '"config"', ['x']

    @pytest.mark.parametrize('vendor,sql,params', [
        ('mysql', 'JSON_TYPE(JSON_EXTRACT("config", %s))', ('x', '$."gain"')),
        ('postgresql', 'JSONB_TYPEOF("config" -> %s)', ('x', 'gain')),
    ])
    def test_vendor_sql(self, vendor, sql, params):
        expr = JSONKeyType('config', 'gain')
        assert getattr(expr, f"as_{vendor}")(self.Compiler(), None) == (sql, params)


@pytest.mark.django_db
class TestTaskListKeywordFilter:
    @pytest.fixture
    def tasks(self, user):
        configs = [
            {'filter': 'R', 'gain': 30, 'time': '2024-05-01 10:00', 'flag': True},
            {'filter': 'r', 'gain': 30.5, 'time': '2023-01-01', 'flag': False},
            {'filter': 'Rc', 'gain': '30', 'target': 'SN 2024abc'},
            {},
        ]
        return [Task.objects.create(original_name=f'img{i}.fits', user=user, config=c)
                for i, c in enumerate(configs)]

    def _ids(self, client, user, tasks, query):
        client.force_login(user)
        resp = client.get(reverse('tasks', kwargs={'id': None}), {'query': query})
        assert resp.status_code == 200
        return [tasks.index(t) for t in sorted(resp.context['tasks'], key=lambda t: t.id)]

    @pytest.mark.parametrize('query,expected', [
        ('filter=R', [0, 1]),
        ('filter!=R', [2]),
        ('gain=30', [0, 2]),
        ('gain>30', [1]),
        ('gain>=30', [0, 1]),
        ('gain<100', [0, 1]),
        ('time>2024-02', [0]),
        # Unquoted numbers are only compared with numeric values
        ('time>2024', []),
        ('time>"2024"', [0]),
        ('time<"2024-01-01"', [1]),
        ('flag=true', [0]),
        ('flag=false', [1]),
        ('target="sn 2024abc"', [2]),
        ('missing=1', []),
        ('filter=R gain>30', [1]),
        ('img1 filter=R', [1]),
        ('contains=1', []),
    ])
    def test_criteria(self, client, user, tasks, query, expected):
        assert self._ids(client, user, tasks, query) == expected

    def test_api_query(self, authenticated_client, tasks):
        response = authenticated_client.get('/api/tasks/', {'query': 'filter=R gain>30'})
        assert response.status_code == 200
        assert [t['id'] for t in response.data] == [tasks[1].id]


@pytest.mark.django_db
class TestTaskListCoordinateFilter:
    def test_point_inside_field(self, client, user):
        """Without explicit radius, the task's own field radius is used."""
        inside = Task.objects.create(original_name='a.fits', user=user, config={}, ra=10.0, dec=20.0, radius=0.5)
        Task.objects.create(original_name='b.fits', user=user, config={}, ra=11.0, dec=20.0, radius=0.5)
        client.force_login(user)
        resp = client.get(reverse('tasks', kwargs={'id': None}), {'query': '10.2 20.0'})
        assert list(resp.context['tasks']) == [inside]

    def test_api_point_with_radius(self, authenticated_client, user):
        Task.objects.create(original_name='a.fits', user=user, config={}, ra=10.0, dec=20.0, radius=0.5)
        far = Task.objects.create(original_name='b.fits', user=user, config={}, ra=11.0, dec=20.0, radius=0.5)
        response = authenticated_client.get('/api/tasks/', {'query': '11.0 20.0 0.1'})
        assert [t['id'] for t in response.data] == [far.id]

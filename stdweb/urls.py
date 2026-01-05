"""
URL configuration for stdweb project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, re_path, include
from django.urls import reverse_lazy
from django.conf import settings
from django.http import HttpResponse

from django.contrib.auth import views as auth_views

from . import views
from . import views_tasks
from . import views_celery
from . import views_skyportal
from . import views_lightcurves

urlpatterns = [
    # REST API
    path('api/', include('stdweb.api.urls')),

    # path('', views.index, name='index'),
    path('', views.upload_file, name='index'),

    path('files/', views.list_files, {'path': ''}, name='files'),
    path('files/<path:path>', views.list_files, name='files'),
    path('view/<path:path>', views.download, {'attachment': False}, name='view'),
    path('download/<path:path>', views.download, {'attachment': True}, name='download'),
    path('preview/<path:path>', views.preview, name='preview'),
    path('cutout/<path:path>', views.cutout, name='cutout'),

    # Documentation
    path('doc/', views.view_markdown, {'path': 'README.md'}, name='doc'),
    path('doc/<path:path>', views.view_markdown, name='doc_file'),

    # Uploads
    path('upload/', views.upload_file, name='upload'),

    # Tasks
    path('tasks/', views_tasks.tasks, {'id':None}, name='tasks'),
    path('tasks/actions', views_tasks.tasks_actions, name='tasks_actions'),
    path('tasks/<int:id>', views_tasks.tasks, name='tasks'),

    path('tasks/<int:id>/files/', views_tasks.task_files, {'path': ''}, name='task_files'),
    path('tasks/<int:id>/files/<path:path>', views_tasks.task_files, name='task_files'),

    path('tasks/<int:id>/preview/<path:path>', views_tasks.task_preview, name='task_preview'),
    path('tasks/<int:id>/view/<path:path>', views_tasks.task_download, {'attachment': False}, name='task_view'),
    path('tasks/<int:id>/download/<path:path>', views_tasks.task_download, {'attachment': True}, name='task_download'),
    path('tasks/<int:id>/cutout/<path:path>', views_tasks.task_cutout, name='task_cutout'),
    path('tasks/<int:id>/candidates_simple', views_tasks.task_candidates, {'filename': 'candidates_simple.vot'}, name='task_candidates_simple'),
    path('tasks/<int:id>/candidates', views_tasks.task_candidates, name='task_candidates'),

    path('tasks/<int:id>/mask/<slug:mode>', views_tasks.task_mask, name='task_mask'),
    path('tasks/<int:id>/state', views_tasks.task_state, name='task_state'),

    path('skyportal/', views_skyportal.skyportal, name='skyportal'),

    # Lightcurves
    path('lightcurves/', views_lightcurves.lightcurves, name='lightcurves'),
    path('lightcurves/photometry/<int:id>/html', views_lightcurves.task_photometry_html, name='lightcurve_photometry_html'),

    # Celery queue
    path('queue/', views_celery.view_queue, {'id': None}, name='queue'),
    path('queue/<slug:id>', views_celery.view_queue, name='queue'),
    path('queue/<slug:id>/state', views_celery.get_queue, name='queue_state'),

    # Auth
    path('login/', auth_views.LoginView.as_view(), name='login'),
    path('logout/', auth_views.LogoutView.as_view(), name='logout'),
    path('password/', auth_views.PasswordChangeView.as_view(success_url=reverse_lazy('password_change_done')), name='password'),
    path('password/done/', auth_views.PasswordChangeDoneView.as_view(), name='password_change_done'),
    path('profile/', views.profile, name='profile'),

    # Action Log (staff only)
    path('log/', views.action_log, name='action_log'),

    # robots.txt
    path('robots.txt', lambda _: HttpResponse("User-agent: *\nDisallow: /\n", content_type="text/plain")),

    # Admin panel
    path('admin/', admin.site.urls),
]

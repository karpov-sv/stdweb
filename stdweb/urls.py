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
from django.conf import settings
from django.http import HttpResponse

from django.contrib.auth import views as auth_views

from . import views
from . import views_tasks
from . import views_celery

urlpatterns = [
    # path('', views.index, name='index'),
    path('', views.upload_file, name='index'),

    path('files/', views.list_files, {'path': ''}, name='files'),
    path('files/<path:path>', views.list_files, name='files'),
    path('view/<path:path>', views.download, {'attachment': False}, name='view'),
    path('download/<path:path>', views.download, {'attachment': True}, name='download'),
    path('preview/<path:path>', views.preview, name='preview'),
    path('cutout/<path:path>', views.cutout, name='cutout'),

    # Uploads
    path('upload/', views.upload_file, name='upload'),
    path('reuse/', views.reuse_file, name='reuse'),

    # Tasks
    path('tasks/', views_tasks.tasks, {'id':None}, name='tasks'),
    path('tasks/<int:id>', views_tasks.tasks, name='tasks'),

    path('tasks/<int:id>/preview/<path:path>', views_tasks.task_preview, name='task_preview'),
    path('tasks/<int:id>/view/<path:path>', views_tasks.task_download, {'attachment': False}, name='task_view'),
    path('tasks/<int:id>/download/<path:path>', views_tasks.task_download, {'attachment': True}, name='task_download'),
    path('tasks/<int:id>/cutout/<path:path>', views_tasks.task_cutout, name='task_cutout'),
    path('tasks/<int:id>/candidates', views_tasks.task_candidates, name='task_candidates'),

    path('tasks/<int:id>/mask', views_tasks.task_mask, name='task_mask'),

    # Celery queue
    path('queue/', views_celery.view_queue, {'id': None}, name='queue'),
    path('queue/<slug:id>', views_celery.view_queue, name='queue'),
    path('queue/<slug:id>/state', views_celery.get_queue, name='queue_state'),

    # Auth
    path('login/', auth_views.LoginView.as_view(), name='login'),
    path('logout/', auth_views.LogoutView.as_view(), name='logout'),

    # robots.txt
    path('robots.txt', lambda r: HttpResponse("User-agent: *\nDisallow: /\n", content_type="text/plain")),
    
    # Admin panel
    path('admin/', admin.site.urls),
]

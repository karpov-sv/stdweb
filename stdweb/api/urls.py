"""
URL configuration for the STDWeb REST API.
"""

from django.urls import path

from rest_framework.authtoken.views import obtain_auth_token

from .views import (
    # Task endpoints
    task_list,
    task_detail,
    task_process,
    task_cancel,
    task_duplicate,
    task_fix,
    task_crop,
    task_destripe,
    task_files_list,
    task_file,
    task_preview,
    task_cutout,
    # Queue endpoints
    queue_list,
    queue_detail,
    queue_terminate,
    # Data file endpoints
    data_files,
    data_file_preview,
    # Preset endpoints
    preset_list,
    preset_detail,
    # Reference data endpoints
    reference_data,
)


urlpatterns = [
    # Token authentication
    path('auth/token/', obtain_auth_token, name='api-token-auth'),

    # Task list and create
    path('tasks/', task_list, name='api-task-list'),

    # Task detail, update, delete
    path('tasks/<int:pk>/', task_detail, name='api-task-detail'),

    # Task actions
    path('tasks/<int:pk>/process/', task_process, name='api-task-process'),
    path('tasks/<int:pk>/cancel/', task_cancel, name='api-task-cancel'),
    path('tasks/<int:pk>/duplicate/', task_duplicate, name='api-task-duplicate'),
    path('tasks/<int:pk>/fix/', task_fix, name='api-task-fix'),
    path('tasks/<int:pk>/crop/', task_crop, name='api-task-crop'),
    path('tasks/<int:pk>/destripe/', task_destripe, name='api-task-destripe'),

    # Task file operations
    path('tasks/<int:pk>/files/', task_files_list, name='api-task-files'),
    path('tasks/<int:pk>/files/<path:path>', task_file, name='api-task-file'),

    # Task preview and cutout
    path('tasks/<int:pk>/preview/<path:path>', task_preview, name='api-task-preview'),
    path('tasks/<int:pk>/cutout/<path:path>', task_cutout, name='api-task-cutout'),

    # Queue management
    path('queue/', queue_list, name='api-queue-list'),
    path('queue/<str:pk>/', queue_detail, name='api-queue-detail'),
    path('queue/<str:pk>/terminate/', queue_terminate, name='api-queue-terminate'),

    # Data files
    path('files/', data_files, name='api-data-files'),
    path('files/<path:path>', data_files, name='api-data-file'),
    path('files/<path:path>/preview/', data_file_preview, name='api-data-file-preview'),

    # Presets
    path('presets/', preset_list, name='api-preset-list'),
    path('presets/<int:pk>/', preset_detail, name='api-preset-detail'),

    # Reference data
    path('reference/', reference_data, name='api-reference'),
    path('reference/<str:data_type>/', reference_data, name='api-reference-detail'),
]

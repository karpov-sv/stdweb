from django.urls import path
from rest_framework.authtoken.views import obtain_auth_token

from . import views_api

app_name = 'api'

urlpatterns = [
    # Authentication
    path('auth/token/', obtain_auth_token, name='api_token_auth'),
    
    # Task endpoints
    path('tasks/upload/', views_api.TaskUploadAPIView.as_view(), name='task_upload'),
    path('tasks/', views_api.task_list_api, name='task_list'),
    path('tasks/<int:task_id>/', views_api.task_detail_api, name='task_detail'),
    path('tasks/<int:task_id>/action/', views_api.task_action_api, name='task_action'),
    path('tasks/<int:task_id>/upload_template/', views_api.task_upload_template_api),
    
    # Preset endpoints
    path('presets/', views_api.preset_list_api, name='preset_list'),
] 
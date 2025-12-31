"""Helper functions for action logging."""

from .models import ActionLog


def log_action(action, user=None, task=None, details=None, request=None):
    """
    Create an action log entry.

    Args:
        action: One of ActionLog.ACTION_TYPES choices
        user: User who performed the action (optional, can be extracted from request)
        task: Task object the action was performed on (optional)
        details: Dict with additional details about the action (optional)
        request: HTTP request object to extract user and IP from (optional)
    """
    ip_address = None

    if request:
        # Extract IP address from request
        ip_address = request.META.get('HTTP_X_FORWARDED_FOR', '').split(',')[0].strip()
        if not ip_address:
            ip_address = request.META.get('REMOTE_ADDR')

        # Extract user from request if not provided
        if not user and hasattr(request, 'user') and request.user.is_authenticated:
            user = request.user

    ActionLog.objects.create(
        user=user,
        action=action,
        task=task,
        task_id_ref=task.id if task else None,
        details=details or {},
        ip_address=ip_address,
    )

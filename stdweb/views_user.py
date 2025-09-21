import os
from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_protect
from django.http import HttpResponse, Http404
from django.conf import settings
from rest_framework.authtoken.models import Token


@login_required
def api_tokens(request):
    """View to manage API tokens for the current user"""
    
    # Get or create token for the user
    token, created = Token.objects.get_or_create(user=request.user)
    
    context = {
        'token': token,
        'token_created': created,
    }
    
    return render(request, 'api_tokens.html', context)


@login_required
@require_http_methods(["POST"])
@csrf_protect
def regenerate_api_token(request):
    """Regenerate API token for the current user"""
    
    try:
        # Delete existing token if it exists
        Token.objects.filter(user=request.user).delete()
        
        # Create new token
        token = Token.objects.create(user=request.user)
        
        messages.success(request, f"New API token generated successfully!")
        
    except Exception as e:
        messages.error(request, f"Error generating API token: {str(e)}")
    
    return redirect('api_tokens')


@login_required
def api_documentation(request):
    """Serve the API documentation markdown file"""
    
    try:
        # Look for API_USAGE.md in the project root
        doc_path = os.path.join(settings.BASE_DIR, 'API_USAGE.md')
        
        if not os.path.exists(doc_path):
            raise Http404("API documentation not found")
        
        with open(doc_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        response = HttpResponse(content, content_type='text/plain; charset=utf-8')
        response['Content-Disposition'] = 'inline; filename="API_USAGE.md"'
        return response
        
    except Exception as e:
        raise Http404(f"Error loading API documentation: {str(e)}") 
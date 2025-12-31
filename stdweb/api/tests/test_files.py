"""
Tests for task file endpoints.
"""

import os
import pytest
from django.core.files.uploadedfile import SimpleUploadedFile
from rest_framework import status


class TestTaskFilesList:
    """Tests for GET /api/tasks/{id}/files/"""

    def test_list_empty_directory(self, authenticated_client, task):
        """Empty directory should return empty list."""
        response = authenticated_client.get(f'/api/tasks/{task.id}/files/')
        assert response.status_code == status.HTTP_200_OK
        assert response.data == []

    def test_list_files(self, authenticated_client, task_with_file):
        """Should list files in task directory."""
        response = authenticated_client.get(f'/api/tasks/{task_with_file.id}/files/')
        assert response.status_code == status.HTTP_200_OK
        assert len(response.data) == 1
        assert response.data[0]['name'] == 'image.fits'
        assert 'size' in response.data[0]
        assert 'modified' in response.data[0]
        assert response.data[0]['is_dir'] is False

    def test_list_other_user_task_denied(self, authenticated_client, other_user_task):
        """Cannot list files of another user's task."""
        response = authenticated_client.get(f'/api/tasks/{other_user_task.id}/files/')
        assert response.status_code == status.HTTP_403_FORBIDDEN

    def test_staff_can_list_any_task(self, staff_client, other_user_task):
        """Staff can list files of any task."""
        response = staff_client.get(f'/api/tasks/{other_user_task.id}/files/')
        assert response.status_code == status.HTTP_200_OK


class TestTaskFileDownload:
    """Tests for GET /api/tasks/{id}/files/{path}"""

    def test_download_file(self, authenticated_client, task_with_file):
        """Can download existing file."""
        response = authenticated_client.get(f'/api/tasks/{task_with_file.id}/files/image.fits')
        assert response.status_code == status.HTTP_200_OK
        assert response.get('Content-Disposition') is not None

    def test_download_nonexistent_file(self, authenticated_client, task):
        """Nonexistent file returns 404."""
        response = authenticated_client.get(f'/api/tasks/{task.id}/files/nonexistent.fits')
        assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_path_traversal_blocked(self, authenticated_client, task):
        """Path traversal attempts should be blocked."""
        response = authenticated_client.get(f'/api/tasks/{task.id}/files/../../../etc/passwd')
        assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_download_other_user_file_denied(self, authenticated_client, other_user_task):
        """Cannot download files from another user's task."""
        # Create a file in other user's task
        filepath = os.path.join(other_user_task.path(), 'secret.txt')
        with open(filepath, 'w') as f:
            f.write('secret content')

        response = authenticated_client.get(f'/api/tasks/{other_user_task.id}/files/secret.txt')
        assert response.status_code == status.HTTP_403_FORBIDDEN


class TestTaskFileUpload:
    """Tests for POST /api/tasks/{id}/files/{path}"""

    def test_upload_file(self, authenticated_client, task):
        """Can upload file to task directory."""
        file_content = b'test file content'
        upload_file = SimpleUploadedFile('test.txt', file_content)

        response = authenticated_client.post(
            f'/api/tasks/{task.id}/files/test.txt',
            {'file': upload_file},
            format='multipart'
        )
        assert response.status_code == status.HTTP_201_CREATED
        assert response.data['path'] == 'test.txt'

        # Verify file was created
        filepath = os.path.join(task.path(), 'test.txt')
        assert os.path.exists(filepath)
        with open(filepath, 'rb') as f:
            assert f.read() == file_content

    def test_upload_to_subdirectory(self, authenticated_client, task):
        """Can upload file to subdirectory."""
        file_content = b'nested content'
        upload_file = SimpleUploadedFile('data.txt', file_content)

        response = authenticated_client.post(
            f'/api/tasks/{task.id}/files/subdir/data.txt',
            {'file': upload_file},
            format='multipart'
        )
        assert response.status_code == status.HTTP_201_CREATED

        # Verify subdirectory was created
        filepath = os.path.join(task.path(), 'subdir', 'data.txt')
        assert os.path.exists(filepath)

    def test_upload_without_file(self, authenticated_client, task):
        """Upload without file returns error."""
        response = authenticated_client.post(f'/api/tasks/{task.id}/files/test.txt', {})
        assert response.status_code == status.HTTP_400_BAD_REQUEST

    def test_upload_to_other_user_task_denied(self, authenticated_client, other_user_task):
        """Cannot upload to another user's task."""
        upload_file = SimpleUploadedFile('malicious.txt', b'hacked')
        response = authenticated_client.post(
            f'/api/tasks/{other_user_task.id}/files/malicious.txt',
            {'file': upload_file},
            format='multipart'
        )
        assert response.status_code == status.HTTP_403_FORBIDDEN

    def test_upload_path_traversal_blocked(self, authenticated_client, task):
        """Path traversal in upload should be blocked."""
        upload_file = SimpleUploadedFile('evil.txt', b'evil content')
        response = authenticated_client.post(
            f'/api/tasks/{task.id}/files/../../../tmp/evil.txt',
            {'file': upload_file},
            format='multipart'
        )
        assert response.status_code == status.HTTP_404_NOT_FOUND


class TestTaskFileDelete:
    """Tests for DELETE /api/tasks/{id}/files/{path}"""

    def test_delete_file(self, authenticated_client, task_with_file):
        """Can delete file from task directory."""
        response = authenticated_client.delete(f'/api/tasks/{task_with_file.id}/files/image.fits')
        assert response.status_code == status.HTTP_204_NO_CONTENT

        filepath = os.path.join(task_with_file.path(), 'image.fits')
        assert not os.path.exists(filepath)

    def test_delete_nonexistent_file(self, authenticated_client, task):
        """Deleting nonexistent file returns 404."""
        response = authenticated_client.delete(f'/api/tasks/{task.id}/files/nonexistent.txt')
        assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_delete_directory(self, authenticated_client, task):
        """Can delete directory."""
        # Create a subdirectory with a file
        subdir = os.path.join(task.path(), 'subdir')
        os.makedirs(subdir)
        with open(os.path.join(subdir, 'file.txt'), 'w') as f:
            f.write('content')

        response = authenticated_client.delete(f'/api/tasks/{task.id}/files/subdir')
        assert response.status_code == status.HTTP_204_NO_CONTENT
        assert not os.path.exists(subdir)

    def test_delete_other_user_file_denied(self, authenticated_client, other_user_task):
        """Cannot delete files from another user's task."""
        # Create a file in other user's task
        filepath = os.path.join(other_user_task.path(), 'important.txt')
        with open(filepath, 'w') as f:
            f.write('important content')

        response = authenticated_client.delete(f'/api/tasks/{other_user_task.id}/files/important.txt')
        assert response.status_code == status.HTTP_403_FORBIDDEN
        assert os.path.exists(filepath)


class TestDataFiles:
    """Tests for GET /api/files/ and /api/files/{path}"""

    def test_list_data_directory(self, authenticated_client, temp_data_path):
        """Can list data directory."""
        # Create a test file
        with open(os.path.join(temp_data_path, 'test.fits'), 'w') as f:
            f.write('test')

        response = authenticated_client.get('/api/files/')
        assert response.status_code == status.HTTP_200_OK
        assert len(response.data) == 1
        assert response.data[0]['name'] == 'test.fits'

    def test_list_subdirectory(self, authenticated_client, temp_data_path):
        """Can list subdirectory."""
        subdir = os.path.join(temp_data_path, 'subdir')
        os.makedirs(subdir)
        with open(os.path.join(subdir, 'nested.fits'), 'w') as f:
            f.write('nested')

        response = authenticated_client.get('/api/files/subdir')
        assert response.status_code == status.HTTP_200_OK
        assert len(response.data) == 1
        assert response.data[0]['name'] == 'nested.fits'

    def test_download_data_file(self, authenticated_client, temp_data_path):
        """Can download data file."""
        content = b'file content'
        with open(os.path.join(temp_data_path, 'download.txt'), 'wb') as f:
            f.write(content)

        response = authenticated_client.get('/api/files/download.txt')
        assert response.status_code == status.HTTP_200_OK

    def test_path_traversal_blocked(self, authenticated_client, temp_data_path):
        """Path traversal should be blocked."""
        response = authenticated_client.get('/api/files/../../../etc/passwd')
        assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_nonexistent_path(self, authenticated_client, temp_data_path):
        """Nonexistent path returns 404."""
        response = authenticated_client.get('/api/files/nonexistent/path')
        assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_unauthenticated_denied(self, api_client, temp_data_path):
        """Unauthenticated access denied."""
        response = api_client.get('/api/files/')
        assert response.status_code == status.HTTP_401_UNAUTHORIZED

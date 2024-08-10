import pytest
import os
from minio import Minio
from tempfile import NamedTemporaryFile

class MinIO:
    def __init__(self, address: str) -> None:
        self.client = Minio(address,
               access_key='minio_access_key',
               secret_key='minio_secret_key',
               secure=False)
    
    def insert(self, bucket_name: str, file_path: str) -> None:
        file_name = file_path.rsplit('/')[-1]
        if not self.client.bucket_exists(bucket_name):
            self.client.make_bucket(bucket_name)
        self.client.fput_object(bucket_name, object_name=file_name, file_path=file_path)
    
    def get(self, bucket_name: str) -> list:
        objects = self.client.list_objects(bucket_name, recursive=True)
        return [obj.object_name for obj in objects]

    def put(self, bucket_name: str, file_path: str) -> None:
        file_name = file_path.rsplit('/')[-1]
        if not self.client.bucket_exists(bucket_name):
            self.client.make_bucket(bucket_name)
        self.client.fput_object(bucket_name, object_name=file_name, file_path=file_path)

    def delete_object(self, bucket_name: str, object_name: str) -> None:
        self.client.remove_object(bucket_name, object_name)

@pytest.fixture
def minio_client():
    address = os.getenv('MINIO_SERVER_URL', '127.0.0.1:9000')
    return MinIO(address)

def test_insert(minio_client):
    bucket_name = 'test-bucket'
    with NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(b'Test content')
        temp_file_path = temp_file.name
    
    minio_client.insert(bucket_name, temp_file_path)
    
    objects = minio_client.get(bucket_name)
    assert os.path.basename(temp_file_path) in objects

    os.remove(temp_file_path)

def test_get(minio_client):
    bucket_name = 'test-bucket'
    objects = minio_client.get(bucket_name)
    assert isinstance(objects, list)

def test_put(minio_client):
    bucket_name = 'test-bucket'
    with NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(b'Test content')
        temp_file_path = temp_file.name
    
    minio_client.put(bucket_name, temp_file_path)
    
    objects = minio_client.get(bucket_name)
    assert os.path.basename(temp_file_path) in objects

    os.remove(temp_file_path)

def test_delete_object(minio_client):
    bucket_name = 'test-bucket'
    with NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(b'Test content')
        temp_file_path = temp_file.name
        file_name = os.path.basename(temp_file_path)
    
    minio_client.insert(bucket_name, temp_file_path)
    
    minio_client.delete_object(bucket_name, file_name)
    
    objects = minio_client.get(bucket_name)
    assert file_name not in objects

    os.remove(temp_file_path)

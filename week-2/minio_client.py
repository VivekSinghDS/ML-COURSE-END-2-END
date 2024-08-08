from typing import List
from minio import Minio

class MinIO:
    def __init__(self, address: str) -> None:
        self.client = Minio(address,
               access_key='minio_access_key',
               secret_key='minio_secret_key',
               secure=False)
    
    def insert(self, bucket_name: str, file_path: str) -> None:
        file_extension = file_path.rsplit('.')[2]
        file_name = file_path.rsplit('/')[1]
        print(file_name, file_extension)
        if not self.client.bucket_exists(bucket_name):
            self.client.make_bucket(bucket_name)
        self.client.fput_object(bucket_name, object_name=file_name, file_path=file_path)
    
    def get(self, bucket_name: str) -> List:
        objects = self.client.list_objects(bucket_name, recursive = True)
        return [object.object_name for object in objects]

    def put(self, bucket_name: str, file_path: str) -> None:
        file_name = file_path.rsplit('/')[-1]
        if not self.client.bucket_exists(bucket_name):
            self.client.make_bucket(bucket_name)
        self.client.fput_object(bucket_name, object_name=file_name, file_path=file_path)

    def delete_object(self, bucket_name: str, object_name: str) -> None:
        object_name = object_name.rsplit('/')[-1]
        self.client.remove_object(bucket_name, object_name)

address = 'localhost:9000'
bucket_name = 'sample23xyz'
sample_file = './sample.txt'
updated_file = './sample2.txt'

client = MinIO(address)

client.insert(bucket_name, sample_file)
client.get(bucket_name)
client.put(bucket_name, updated_file)
client.get(bucket_name)
# client.delete_object(bucket_name, sample_file)
# client.get(bucket_name)

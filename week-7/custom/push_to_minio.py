from minio import Minio
import os

# MinIO configuration
minio_url = "http://127.0.0.1:57445"  # Use the MinIO service URL provided by `minikube service minio --url`
access_key = "minio_access_key"  # Replace with your access key
secret_key = "minio_secret_key"  # Replace with your secret key

# Initialize MinIO client
minio_client = Minio(
    minio_url.replace("http://", "").replace("https://", ""),  # Remove protocol from the URL
    access_key=access_key,
    secret_key=secret_key,
    secure=False  # Set to True if you're using HTTPS
)

# Name of the bucket where data will be uploaded
bucket_name = "ml-models"

# Create bucket if it doesn't exist
if not minio_client.bucket_exists(bucket_name):
    minio_client.make_bucket(bucket_name)
    print(f"Bucket '{bucket_name}' created successfully!")
else:
    print(f"Bucket '{bucket_name}' already exists.")

# Define the local path to the data you want to upload (model checkpoints, etc.)
local_directory_path = "/Users/vivek.singh/ML-COURSE/week-7/custom/checkpoint-350"
minio_dir_prefix = "week-7/custom/checkpoint-350" 


def upload_directory_to_minio(local_dir, minio_prefix):
    for root, dirs, files in os.walk(local_dir):
        for file_name in files:
            # Construct full local file path
            local_file_path = os.path.join(root, file_name)
            
            # Construct MinIO object name
            relative_path = os.path.relpath(local_file_path, local_dir)
            object_name = f"{minio_prefix}/{relative_path}"
            
            # Upload the file to MinIO
            try:
                minio_client.fput_object(bucket_name, object_name, local_file_path)
                print(f"File '{object_name}' uploaded successfully!")
            except Exception as e:
                print(f"Error uploading file '{object_name}': {str(e)}")

# Call the function to upload the directory
upload_directory_to_minio(local_directory_path, minio_dir_prefix)

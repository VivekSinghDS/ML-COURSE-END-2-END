# ML-COURSE-END-2-END

The following course will cover a sample project in its entireity. The design 
doc

## Get MinIO via Docker

```
docker run -d \
  -p 9000:9000 \
  -p 9001:9001 \
  --name minio \
  -v ~/minio/data:/data \
  -e "MINIO_ROOT_USER=minio_access_key" \
  -e "MINIO_ROOT_PASSWORD=minio_secret_key" \
  quay.io/minio/minio server /data --console-address ":9001"
```

## Get MinIO via Minikube
```
kubectl apply -f ./week-2/k8s_config/deployment.yaml
kubectl apply -f ./week-2/k8s_config/service.yaml
minikube service minio --url
```

## Multiprocessing Performance Analysis

This experiment compares the execution time and speedup achieved by using multiple processes versus a single process.
The file is present in `week-2/single_multiprocessing.py`

## Results

- **Single-process time:** 21.2082 seconds
- **2-process time:** 10.7991 seconds
  - **Speedup:** 1.96x
- **4-process time:** 6.3731 seconds
  - **Speedup:** 3.33x
- **8-process time:** 3.1998 seconds
  - **Speedup:** 6.63x


The results demonstrate significant performance gains with multiprocessing. As the number of processes increases, the execution time decreases, achieving up to a 6.63x speedup with 8 processes. This highlights the effectiveness of parallel processing in reducing computation time.

## Use DVC + MinIO
```
dvc init 
dvc add ./week-3/sample.json 
dvc remote add -d minio s3://dvc-store
dvc remote modify minio endpointurl http://127.0.0.1:53132
dvc remote modify minio access_key_id minio_access_key
dvc remote modify minio secret_access_key minio_secret_key
dvc push
```
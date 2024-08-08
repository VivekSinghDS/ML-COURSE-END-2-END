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
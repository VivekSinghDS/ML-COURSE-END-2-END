FROM public.ecr.aws/docker/library/python:3.9.10

RUN useradd -ms /bin/bash appuser
WORKDIR /home/appuser

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

USER appuser
EXPOSE 8000
COPY --chown=appuser:appuser . .

CMD ["uvicorn", "sample-web-server:app", "--port", "8000", "--host", "0.0.0.0", "--no-access-log"]
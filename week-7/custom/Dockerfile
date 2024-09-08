FROM python:3.11-slim

RUN useradd -ms /bin/bash appuser
WORKDIR /home/appuser

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt
RUN pip install uvicorn fastapi
ENV HUGGING_FACE_HUB_TOKEN="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"
COPY api.py api.py
# Change ownership of the working directory to the appuser
RUN chown -R appuser:appuser /home/appuser

# Switch to the non-root user
USER appuser

# Set a writable cache directory under the user's home directory
ENV XDG_CACHE_HOME=/home/appuser/.cache
RUN mkdir -p /home/appuser/.cache && chmod -R 777 /home/appuser/.cache

# Expose the application port
EXPOSE 6000

# Run the app with uvicorn
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "6000"]
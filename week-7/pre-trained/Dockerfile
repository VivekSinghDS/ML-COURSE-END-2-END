FROM python:3.9-slim

WORKDIR /app

COPY api.py /app/

RUN pip install fastapi uvicorn requests python-dotenv

ENV HUGGING_FACE_HUB_TOKEN="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"

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "5000"]

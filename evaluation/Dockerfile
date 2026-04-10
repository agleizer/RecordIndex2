FROM python:3.11-slim

WORKDIR /app

COPY evaluation/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY evaluation/ .

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8001"]

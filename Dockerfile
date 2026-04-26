FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN grep -v '^torch' requirements.txt > requirements.docker.txt \
    && python -m pip install --no-cache-dir -r requirements.docker.txt \
    && python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.11.0

COPY app ./app
COPY Makefile .

ENV PYTHONUNBUFFERED=1

RUN python -m app.cli train

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

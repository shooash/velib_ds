FROM tensorflow/tensorflow:2.19.0-gpu
#FROM python:3.12-slim

WORKDIR /app

COPY requirements-cuda.txt .


ARG USE_CUDA=true
RUN pip install --upgrade pip
RUN pip freeze > /tmp/constraints.txt
RUN pip install --no-cache-dir -r requirements-cuda.txt -c /tmp/constraints.txt;

# Copy source code and local data/models for api
# COPY ../../src/ /app/src/
# COPY ../../local/ /app/local/

EXPOSE 8000

# CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]


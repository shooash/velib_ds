FROM tensorflow/tensorflow:2.15.0-gpu
#FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
COPY requirements-cuda.txt .


ARG USE_CUDA=false
RUN pip freeze > /tmp/constraints.txt
RUN if [ "$USE_CUDA" = "true" ]; then \
        echo "Installing CUDA requirements"; \
        pip install --no-cache-dir -r requirements-cuda.txt -c /tmp/constraints.txt; \
    else \ 
        echo "No CUDA requirements installation"; \
        pip install --no-cache-dir -r requirements.txt -c /tmp/constraints.txt; \    
    fi

# Copy source code and local data/models for api
COPY flows/src/ /app/src/
COPY flows/local/ /app/local/

EXPOSE 8000

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]


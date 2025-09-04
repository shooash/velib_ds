FROM tensorflow/tensorflow:2.19.0-gpu

WORKDIR /app

COPY requirements-cuda.txt .
COPY requirements.txt .

ARG USE_CUDA=true
RUN pip install --upgrade pip
RUN pip freeze > /tmp/constraints.txt
RUN if [ "$USE_CUDA" = "true" ]; then \
        echo "Installing CUDA requirements"; \
        pip install --no-cache-dir -r requirements-cuda.txt -c /tmp/constraints.txt; \
    else \ 
        echo "No CUDA requirements installation"; \
        pip install --no-cache-dir -r requirements.txt -c /tmp/constraints.txt; \    
    fi

EXPOSE 8000



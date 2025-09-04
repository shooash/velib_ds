# Velib Airflow Scheduling

This directory contains Apache Airflow configuration for scheduling Velib data refresh and model retraining operations.

## Architecture

The Airflow setup includes:

- **Web Server**: Airflow UI accessible at http://localhost:8080
- **Scheduler**: Manages DAG execution and scheduling
- **PostgreSQL**: Database for Airflow metadata
- **DAGs**: Workflow definitions for data refresh and model retraining

## DAGs

### 1. velib_data_refresh
- **Schedule**: Every 6 hours (`0 */6 * * *`)
- **Purpose**: Refreshes Velib data from external sources
- **Tasks**: Health check → Data refresh

### 2. velib_model_retrain
- **Schedule**: Every Sunday at 2 AM (`0 2 * * 0`)
- **Purpose**: Retrains the ML model with latest data
- **Tasks**: Health check → Model retrain

### 3. velib_refresh_and_retrain
- **Schedule**: Every Sunday at 1 AM (`0 1 * * 0`)
- **Purpose**: Combined workflow for data refresh followed by model retraining
- **Tasks**: Health check → Data refresh → Model retrain

## Setup Instructions

1. Copy `env.example` to `.env` and configure:
   ```bash
   cp env.example .env
   ```

2. Set Airflow user ID (Linux/Mac):
   ```bash
   echo -e "AIRFLOW_UID=$(id -u)" >> .env
   ```

3. Start all services:
   ```bash
   docker-compose up -d
   ```

4. Access Airflow UI:
   - URL: http://localhost:8080
   - Username: airflow
   - Password: airflow

## Monitoring

- **Airflow UI**: http://localhost:8080
- **FastAPI Health**: http://localhost:8001/health



### Environment Variables

Key environment variables:
- `VELIB_FASTAPI_URL`: FastAPI service URL
- `VELIB_RETRAIN_ENDPOINT`: Retrain endpoint path
- `VELIB_REFRESH_ENDPOINT`: Refresh endpoint path

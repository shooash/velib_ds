from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.http_operator import SimpleHttpOperator
from airflow.operators.python_operator import PythonOperator
from airflow.utils.dates import days_ago
import requests
import logging

default_args = {
    'owner': 'velib-team',
    'depends_on_past': False,
    'start_date': days_ago(1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=10),
}

dag = DAG(
    'velib_model_retrain',
    default_args=default_args,
    description='Retrain Velib ML model with latest data',
    schedule_interval='0 14 * * 6',  # Every Saturday at 2 PM
    catchup=False,
    tags=['velib', 'ml', 'retrain'],
)

def check_fastapi_health():
    """Check if FastAPI service is healthy before making requests"""
    try:
        response = requests.get('http://fastapi_big:8000/health', timeout=10)
        if response.status_code == 200:
            logging.info("FastAPI service is healthy")
            return True
        else:
            logging.error(f"FastAPI service returned status {response.status_code}")
            return False
    except Exception as e:
        logging.error(f"Failed to connect to FastAPI service: {str(e)}")
        return False

health_check = PythonOperator(
    task_id='check_fastapi_health',
    python_callable=check_fastapi_health,
    dag=dag,
)

# Task to retrain model
retrain_model = SimpleHttpOperator(
    task_id='retrain_model',
    http_conn_id='velib_fastapi',
    endpoint='/admin/retrain',
    method='POST',
    headers={'Content-Type': 'application/json'},
    dag=dag,
)

health_check >> retrain_model

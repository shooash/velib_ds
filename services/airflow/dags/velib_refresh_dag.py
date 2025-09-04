from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.http_operator import SimpleHttpOperator
from airflow.operators.python_operator import PythonOperator
from airflow.utils.dates import days_ago
import requests
import logging

# Default arguments for the DAG
default_args = {
    'owner': 'velib-team',
    'depends_on_past': False,
    'start_date': days_ago(1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'velib_data_refresh',
    default_args=default_args,
    description='Refresh Velib data from external sources',
    schedule_interval='0 */6 * * *', 
    catchup=False,
    tags=['velib', 'data', 'refresh'],
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

refresh_data = SimpleHttpOperator(
    task_id='refresh_data',
    http_conn_id='velib_fastapi',
    endpoint='/admin/refresh',
    method='POST',
    headers={'Content-Type': 'application/json'},
    dag=dag,
)

health_check >> refresh_data

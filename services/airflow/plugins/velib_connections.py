"""
Airflow plugin to define HTTP connections for Velib FastAPI service
"""
from airflow.plugins_manager import AirflowPlugin
from airflow.models import Connection
from airflow.utils.db import provide_session
from airflow import settings

@provide_session
def create_velib_connection(session=None):
    """Create HTTP connection for Velib FastAPI service"""
    conn_id = 'velib_fastapi'
    
    # Check if connection already exists
    existing_conn = session.query(Connection).filter(Connection.conn_id == conn_id).first()
    
    if not existing_conn:
        new_conn = Connection(
            conn_id=conn_id,
            conn_type='http',
            host='fastapi_big',
            port=8000,
            schema='http',
            description='Velib FastAPI service connection'
        )
        session.add(new_conn)
        session.commit()
        print(f"Created connection: {conn_id}")
    else:
        print(f"Connection {conn_id} already exists")

class VelibConnectionsPlugin(AirflowPlugin):
    name = "velib_connections"
    hooks = []
    executors = []
    macros = []
    admin_views = []
    flask_blueprints = []
    menu_links = []
    appbuilder_views = []
    appbuilder_menu_items = []

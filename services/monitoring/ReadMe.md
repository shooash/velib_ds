# Velib DS Monitoring Services

## Setup
1. Build docker containers
```shell
cd services/monitoring
docker compose build
```
2. Setup Grafana
- Use a password from services/monitoring/grafana/config.monitoring environment variable GF_SECURITY_ADMIN_PASSWORD to access Grafana server (default port: http://localhost:3000/)
- Go to Dashboards - New - Import and import all Grafana Velib Dashboards from /data/grafana/dashboards

## Run
```shell
cd services/monitoring
docker compose up
```

## Services

### MLFlow: 
**Container**: mlflow\
**Description**: Machine Learning lifecycle platform to track models and data continuous development.\
**Port**: 8080

### Prometheus
**Container**: velib_exporter\
**Description**: Service to reveal MLFlow and data metrics for Prometheus\
**Port**: 5000

**Container**: sql_exporter\
**Description**: Service to load and expose Google Cloud Platform service and data status\
**Port**: 9399

**Container**: alertmanager\
**Description**: Alert Manager for Prometheus\
**Port**: 9093

**Container**: prometheus\
**Description**: Prometheus monitoring server\
**Port**: 9090

### Graphana
**Container**: graphana\
**Description**: Graphana dashboarding service to visualize monitoring data\
**Port**: 3000

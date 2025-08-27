# Velib DS Monitoring Services

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

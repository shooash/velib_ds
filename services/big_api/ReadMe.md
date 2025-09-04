# BIG API
Access point for "heavy" calls to Velib DS Project servers: retrain models.

## Setup
*Attention: This image has different build options for NVidia and non-NVidia machines.*

If your machine doesn't have NVidia GPU you can desactivate NVidia CUDA drivers by editing **cuda.env** file:
```shell
USE_CUDA = false
```
You can prebuild image to facilitate deployement:
```shell
docker compose build fastapi_big
```
## Usage
Start server with docker compose command
```shell
docker compose up
```
### Load Newest Data to Server
This would fetch the data from the last timestamp of the dataframe stored on server untill the last available timestamp stored on Google Cloud Platform (or the whole dataset if no data is available in local folder).
Processed datasets are saved to /local/data folder.
```shell
curl -X 'POST' \
  'http://localhost:8001/admin/refresh' \
  -H 'accept: application/json' \
  -d ''
```

### Retrain model
The model is trained using the data preloaded to server. The dataset is transformed following the configuration file /local/configs/transformer_params.json. If the file doesn't exist, default params from /data/transformer_params.default.json are used.
A new Keras mlp model is trained using parameters from a configuration file /local/configs/best_params.json (or /data/best_params.default.json).
During this stage a stations-clusters dataset is saved to /local/data and models (VelibTransformer and MLPFlow) are saved to /local/models to be used for predictions.
```shell
curl -X 'POST' \
  'http://localhost:8001/admin/retrain' \
  -H 'accept: application/json' \
  -d ''
```

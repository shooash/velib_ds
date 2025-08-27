# Tensorflow Py API
Access point for "heavy" calls to Velib DS Project servers: retrain models.

## Setup
*Attention: This image has different build options for NVidia and non-NVidia machines.*

If your machine has NVidia GPU you can activate NVidia CUDA drivers by editing **.env** file:
```shell
USE_CUDA = true
```
You can prebuild image to facilitate deployement:
```shell
docker compose build fastapi
```
## Usage
Start server with docker compose command
```shell
docker compose up
```
TODO: Request model retraining with current data:
```shell
curl
```
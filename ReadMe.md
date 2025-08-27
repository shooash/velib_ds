# Setup

# Testing
## Description
Testing cases:
- **test_velibdata_extract_new** \
    Data extraction from GCP and MeteoFrance API (VelibData, VelibConnector, MeteoFranceConnector).
- **test_velibdata_use_cache**\
    Data loading from cache (VelibData).
- **test_velib_transform** (depends on: test_velibdata_use_cache) \
    ETL data transformation (VelibData)
- **test_transformer_smoothen** (depends on: test_velib_transform)\
    Target value smoothening (VelibTransformer)
- **test_transformer_split** (depends on: test_velib_transform)\
    Data split by date.
- **test_transformer_fit** (depends on: test_transformer_split)\
    Fitting VelibTransformer (VelibTransformer, VelibLagger, VelibClusterizer)
- **test_transformer_transform** (depends on test_transformer_fit)\
    Data transformation and feature engineering (VelibTransformer, VelibLagger, VelibClusterizer)
## Run tests
### With docker
```shell
docker compose up testing
```

### From Python environment
```shell
pytest -x src/test.py
```

# Running
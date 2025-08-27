import os

APP_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), r'../..'))
LOCAL_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), r'../../local'))
LOGS_FOLDER = LOCAL_FOLDER + '/logs'
MODELS_FOLDER = LOCAL_FOLDER + '/models'
PARAMS_FOLDER = LOCAL_FOLDER + '/params'
DATA_FOLDER = LOCAL_FOLDER + '/data'
STATS_FOLDER = DATA_FOLDER + '/stats'


# Dataset extracted from GCP and MétéoFrance, 
# cleaned from extreme outliers, with reconstructed 
# target data and primary feature ingineering (week, month, holiday, vacances etc.)
EXTRACTED_DATASET = LOCAL_FOLDER + r'/data/reconstructed_dataset7.h5'
EXTRACTED_CLUSTERS = LOCAL_FOLDER + r'/data/reconstructed_dataset_clusters.h5'

# Full dataset after additional feature ingeneering (Transfromer.fit(df))
FULL_DATASET = LOCAL_FOLDER + r"/data/prod_dataset.h5"
# Train dataset after additional feature ingeneering (Transfromer.fit(df))
TRAIN_DATASET = LOCAL_FOLDER + r"/data/test_dataset_train.h5"
# Test dataset after additional feature ingeneering (Transfromer.fit(df))
TEST_DATASET = LOCAL_FOLDER + r"/data/test_dataset_test.h5"
# Dataset for lagged feature ingeneering for predictions
PRED_DATASET = LOCAL_FOLDER + r"/data/prod_dataset_pred.h5"

# Transformer object for train test procedures
TRANSFORMER_TRAIN_TEST = LOCAL_FOLDER + r"/data/test_transformer.pkl"
# Transformer object fitted with FULL_DATASET
TRANSFORMER_RELEASE = LOCAL_FOLDER + r"/data/release_transformer.pkl"
# Transformer object fitted with FULL_DATASET and used to transform it (with data to scale/transform X for predictions).
TRANSFORMER_PROD = LOCAL_FOLDER + r"/data/prod_transformer.pkl"

# MLP Best run
MLP_BEST_RUN = LOCAL_FOLDER + r'/params/best.run.txt'
# MLP Series Best run
MLP_SERIES_BEST_RUN = LOCAL_FOLDER + r'/params/series_best.run.txt'

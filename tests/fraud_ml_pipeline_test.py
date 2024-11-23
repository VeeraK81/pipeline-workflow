# import pytest
# from unittest import mock
# from app.fraud_detection_train import load_data, preprocess_data, create_pipeline, train_model
# from io import StringIO
# import os


# # Test data loading
# def test_load_data():
#     # S3 Bucket and Key
#     df = load_data()
 
#     assert not df.empty, "Dataframe is empty"

# # Test data preprocessing
# def test_preprocess_data():
#     df = load_data()
#     X_train, X_test, y_train, y_test = preprocess_data(df)
#     assert len(X_train) > 0, "Training data is empty"
#     assert len(X_test) > 0, "Test data is empty"

# # Test pipeline creation
# def test_create_pipeline():
#     pipe = create_pipeline()
#     assert "scaler" in pipe.named_steps, "Scaler missing in pipeline"
#     assert "classifier" in pipe.named_steps, "Classifier missing in pipeline"

# # Test model training (mocking GridSearchCV)
# # @mock.patch('app.train.GridSearchCV.fit', return_value=None)
# @mock.patch('app.fraud_detection_train.GridSearchCV.fit', return_value=None)
# def test_train_model(mock_fit):
#     pipe = create_pipeline()
#     X_train, X_test, y_train, y_test = preprocess_data(load_data())
#     param_grid = {
#         "classifier__n_estimators": [100, 150],
#         "classifier__max_depth": [10, 20, 30],
#         "classifier__min_samples_split": [2, 5],
#         "classifier__min_samples_leaf": [1, 2]
#     }   
#     model = train_model(pipe, X_train, y_train, param_grid)
#     assert model is not None, "Model training failed"

import pytest
from unittest import mock
from app.fraud_detection_train import load_data, preprocess_data, create_pipeline, train_model
from io import StringIO
import os

# Test data loading
def test_load_data():
    # S3 Bucket and Key
    df = load_data()
    assert not df.empty, "Dataframe is empty"

# Test data preprocessing
def test_preprocess_data():
    df = load_data()
    X_train, X_test, y_train, y_test = preprocess_data(df)
    assert len(X_train) > 0, "Training data is empty"
    assert len(X_test) > 0, "Test data is empty"

# Test pipeline creation
def test_create_pipeline():
    pipelines = create_pipeline()
    assert isinstance(pipelines, list), "Pipelines should be a list"
    assert len(pipelines) > 0, "No pipelines created"

    for name, pipe in pipelines:
        assert "scaler" in pipe.named_steps, f"Scaler missing in pipeline: {name}"
        assert "classifier" in pipe.named_steps, f"Classifier missing in pipeline: {name}"

# Test model training (mocking GridSearchCV)
@mock.patch('app.fraud_detection_train.GridSearchCV.fit', return_value=None)
def test_train_model(mock_fit):
    pipelines = create_pipeline()
    X_train, X_test, y_train, y_test = preprocess_data(load_data())

    # Define the parameter grid for both models
    param_grids = {
        "RandomForest": {
            "classifier__n_estimators": [100, 150],
            "classifier__max_depth": [10, 20, 30],
            "classifier__min_samples_split": [2, 5],
            "classifier__min_samples_leaf": [1, 2]
        },
        "LogisticRegression": {
            "classifier__C": [0.1, 1.0, 10],  # Regularization strength
            "classifier__penalty": ['l2'],   # Regularization type
            "classifier__solver": ['lbfgs', 'liblinear']  # Solvers for optimization
        }
    }
    
    # Test each pipeline
    for name, pipe in pipelines:
        model = train_model([(name, pipe)], X_train, y_train, {name: param_grids[name]})
        assert model is not None, f"Model training failed for {name}"

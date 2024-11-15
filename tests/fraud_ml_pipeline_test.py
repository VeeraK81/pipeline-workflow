import pytest
from unittest import mock
from app.train import load_data, preprocess_data, create_pipeline, train_model

s3_url = "s3://mymlflowbuc/transactions/fraudTest.csv"

# Test data loading
def test_load_data():
    # url = "https://julie-2-next-resources.s3.eu-west-3.amazonaws.com/full-stack-full-time/linear-regression-ft/californian-housing-market-ft/california_housing_market.csv"
    url = s3_url
    df = load_data(url)
    assert not df.empty, "Dataframe is empty"

# Test data preprocessing
def test_preprocess_data():
    # df = load_data("https://julie-2-next-resources.s3.eu-west-3.amazonaws.com/full-stack-full-time/linear-regression-ft/californian-housing-market-ft/california_housing_market.csv")
    df = load_data(s3_url)
    X_train, X_test, y_train, y_test = preprocess_data(df)
    assert len(X_train) > 0, "Training data is empty"
    assert len(X_test) > 0, "Test data is empty"

# Test pipeline creation
def test_create_pipeline():
    pipe = create_pipeline()
    assert "scaler" in pipe.named_steps, "Scaler missing in pipeline"
    assert "classifier" in pipe.named_steps, "Classifier missing in pipeline"

# Test model training (mocking GridSearchCV)
# @mock.patch('app.train.GridSearchCV.fit', return_value=None)
@mock.patch('app.fraud_detection_train.GridSearchCV.fit', return_value=None)
def test_train_model(mock_fit):
    pipe = create_pipeline()
    # X_train, X_test, y_train, y_test = preprocess_data(load_data("https://julie-2-next-resources.s3.eu-west-3.amazonaws.com/full-stack-full-time/linear-regression-ft/californian-housing-market-ft/california_housing_market.csv"))
    X_train, X_test, y_train, y_test = preprocess_data(load_data(s3_url))
    # param_grid = {"Random_Forest__n_estimators": [90], "Random_Forest__criterion": ["squared_error"]}
    param_grid = {
        "classifier__n_estimators": [100, 150],
        "classifier__learning_rate": [0.01, 0.1],
        "classifier__max_depth": [3, 5]
    }
    model = train_model(pipe, X_train, y_train, param_grid)
    assert model is not None, "Model training failed"

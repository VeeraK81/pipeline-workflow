import pytest
from unittest import mock
from app.fraud_detection_train import load_data, preprocess_data, create_pipeline, train_model
from io import StringIO
import os
import time
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


# Test data loading
def test_load_data():
    # Test the load_data function to check if it correctly loads the data from S3
    df = load_data()
    
    # Ensure the DataFrame is not empty after loading the data
    assert not df.empty, "Dataframe is empty"


# Test data preprocessing
def test_preprocess_data():
    # Load data and preprocess it
    df = load_data()
    X_train, X_test, y_train, y_test = preprocess_data(df)
    
    # Ensure that the training data is not empty
    assert len(X_train) > 0, "Training data is empty"
    
    # Ensure that the test data is not empty
    assert len(X_test) > 0, "Test data is empty"


# Test pipeline creation
def test_create_pipeline():
    # Test the create_pipeline function to check if the pipelines are created properly
    pipelines = create_pipeline()
    
    # Check if the returned pipelines are in a list format
    assert isinstance(pipelines, list), "Expected a list of pipelines"  
    
    # Iterate over each pipeline and perform checks
    for name, pipe in pipelines:
        # Check that each pipeline is a tuple with a name and a Pipeline object
        assert isinstance(name, str), "Pipeline name should be a string"
        assert isinstance(pipe, Pipeline), f"Expected a Pipeline object, got {type(pipe)}"
        
        # Check that the pipeline contains the necessary 'scaler' and 'classifier' steps
        assert "scaler" in pipe.named_steps, f"Scaler missing in pipeline '{name}'"
        assert "classifier" in pipe.named_steps, f"Classifier missing in pipeline '{name}'"

        # Optional: Check that the classifier type matches the expected one
        if name == "RandomForest":
            assert isinstance(pipe.named_steps["classifier"], RandomForestClassifier), "Expected RandomForestClassifier"
        elif name == "LogisticRegression":
            assert isinstance(pipe.named_steps["classifier"], LogisticRegression), "Expected LogisticRegression"


# Test model training (mocking GridSearchCV)
def test_train_model():
    # Load and preprocess data
    X_train, X_test, y_train, y_test = preprocess_data(load_data())
    
    # Define the parameter grid for both models (RandomForest and LogisticRegression)
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

    # Create the model pipelines
    pipelines = create_pipeline()
    
    # Iterate over each pipeline and train models
    for name, pipe in pipelines:
        print(f"Training with pipeline: {name}")
        
        # Train model with GridSearchCV and the corresponding parameter grid
        model = train_model([(name, pipe)], X_train, y_train, {name: param_grids[name]})
        
        # Ensure the model is returned after training
        assert model is not None, f"Model training failed for {name}"

        # Check that the best model has the 'best_score_' attribute (indicating successful training)
        assert hasattr(model, 'best_score_'), f"Model for {name} does not have a 'best_score_' attribute."
        
        # Ensure that the best model also has the 'best_params_' attribute (indicating hyperparameters were tuned)
        assert hasattr(model, 'best_params_'), f"Model for {name} does not have 'best_params_' attribute."

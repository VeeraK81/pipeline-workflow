import pandas as pd
import numpy as np
import mlflow
import time
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, log_loss, roc_auc_score
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import boto3
import os
from io import StringIO


# Function to load data from an S3 bucket
def load_data():
    bucket_name = os.getenv('BUCKET_NAME')  # Get bucket name from environment variables
    file_key = os.getenv('FILE_KEY')  # Get file key (file path) from environment variables

    # Create an S3 client using the AWS credentials
    s3_client = boto3.client(
        's3',
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),  # AWS Access Key
        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')  # AWS Secret Key
    )

    # Fetch the CSV file from the S3 bucket
    response = s3_client.get_object(Bucket=bucket_name, Key=file_key)
    csv_content = response['Body'].read().decode('utf-8')  # Decode the file content to a string
    
    # Load the CSV content into a pandas DataFrame
    return pd.read_csv(StringIO(csv_content))

# Function to preprocess the data
def preprocess_data(df):
    # Create a new feature: distance to the merchant
    df['distance_to_merchant'] = np.sqrt((df['lat'] - df['merch_lat'])**2 + (df['long'] - df['merch_long'])**2)
    
    # Extract day of week and hour from the transaction date-time
    df['trans_dayofweek'] = pd.to_datetime(df['trans_date_trans_time']).dt.dayofweek
    df['trans_hour'] = pd.to_datetime(df['trans_date_trans_time']).dt.hour
    
    # Define features and target for the model
    features = ['amt', 'distance_to_merchant', 'city_pop', 'trans_dayofweek', 'trans_hour']
    X = df[features]  # Feature set
    y = df['is_fraud']  # Target variable (fraud detection)
    
    # Split the data into training and test sets (80% train, 20% test)
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Function to create the model pipeline for both RandomForest and LogisticRegression
def create_pipeline():
    # Creating a list of tuples, each containing a pipeline for a different model
    return [
        ("RandomForest", Pipeline(steps=[  # Pipeline for Random Forest
            ("scaler", StandardScaler()),  # Feature scaling
            ("classifier", RandomForestClassifier(random_state=42))  # Random Forest classifier
        ])),
        ("LogisticRegression", Pipeline(steps=[  # Pipeline for Logistic Regression
            ("scaler", StandardScaler()),  # Feature scaling
            ("classifier", LogisticRegression(random_state=42, max_iter=1000))  # Logistic Regression classifier
        ]))
    ]

# Function to train the model with hyperparameter tuning using GridSearchCV
def train_model(pipelines, X_train, y_train, param_grids, cv=2, n_jobs=-1, verbose=3):
    best_model = None
    best_score = 0
    best_pipeline = None

    # Iterate over each pipeline to train the models
    for name, pipe in pipelines:
        print(f"Training with pipeline: {name}")
        model = GridSearchCV(pipe, param_grids[name], cv=cv, n_jobs=n_jobs, verbose=verbose, scoring="accuracy")
        model.fit(X_train, y_train)  # Fit the model to the training data

        # If this model performs better than previous models, update the best model
        if model.best_score_ > best_score:
            best_score = model.best_score_
            best_model = model
            best_pipeline = name

    print(f"Best model selected: {best_pipeline} with score: {best_score}")
    return best_model

# Function to log the metrics and model to MLflow
def log_metrics_and_model(model, X_train, y_train, X_test, y_test, artifact_path, registered_model_name):
    # Log the training and testing accuracy
    train_score = accuracy_score(y_train, model.predict(X_train))
    test_score = accuracy_score(y_test, model.predict(X_test))
    mlflow.log_metric("Train Accuracy", train_score)
    mlflow.log_metric("Test Accuracy", test_score)

    # Log the trained model to MLflow artifacts
    mlflow.sklearn.log_model(
        sk_model=model.best_estimator_,  # Log the best model from grid search
        artifact_path=artifact_path
    )

    # Construct model URI for registration
    model_uri = f"runs:/{mlflow.active_run().info.run_id}/{artifact_path}"
    
    # Debugging output
    print(f"Model URI for registration: {model_uri}")

    # Attempt to register the model in MLflow's model registry
    try:
        result = mlflow.register_model(model_uri=model_uri, name=registered_model_name)
        print(f"Model registered with version: {result.version}")
    except Exception as e:
        print(f"Error registering model: {str(e)}")

# Function to log additional metrics such as F1, Precision, Recall, Log Loss, and ROC AUC
def log_additional_metrics(model, X_train, y_train, X_test, y_test):
    # Calculate additional metrics for training data
    train_accuracy = accuracy_score(y_train, model.predict(X_train))
    train_f1 = f1_score(y_train, model.predict(X_train))
    train_precision = precision_score(y_train, model.predict(X_train))
    train_recall = recall_score(y_train, model.predict(X_train))
    train_log_loss = log_loss(y_train, model.predict_proba(X_train))
    train_roc_auc = roc_auc_score(y_train, model.predict_proba(X_train)[:, 1])

    # Log metrics to MLflow
    mlflow.log_metric("training_accuracy_score", train_accuracy)
    mlflow.log_metric("training_f1_score", train_f1)
    mlflow.log_metric("training_precision_score", train_precision)
    mlflow.log_metric("training_recall_score", train_recall)
    mlflow.log_metric("training_log_loss", train_log_loss)
    mlflow.log_metric("training_roc_auc", train_roc_auc)

    # Log the best cross-validation score from GridSearchCV
    mlflow.log_metric("best_cv_score", model.best_score_)

# Main function to run the experiment
def run_experiment(experiment_name, param_grids, artifact_path, registered_model_name):
    start_time = time.time()  # Record the start time

    # Load and preprocess the data
    df = load_data()
    X_train, X_test, y_train, y_test = preprocess_data(df)

    # Create the model pipelines
    pipelines = create_pipeline()

    # Set the experiment's info in MLflow
    mlflow.set_experiment(experiment_name)
    experiment = mlflow.get_experiment_by_name(experiment_name)

    # Ensure no active runs are left open
    if mlflow.active_run():
        mlflow.end_run()
        
    # Enable MLflow autologging for automatic logging of hyperparameters, metrics, etc.
    mlflow.sklearn.autolog()

    # Start a new MLflow run
    with mlflow.start_run(experiment_id=experiment.experiment_id):
        # Train the model
        model = train_model(pipelines, X_train, y_train, param_grids)

        # Log metrics and model to MLflow
        log_metrics_and_model(model, X_train, y_train, X_test, y_test, artifact_path, registered_model_name)
        
        # Log additional metrics
        log_additional_metrics(model, X_train, y_train, X_test, y_test)
        
    # Print the total training time
    print(f"...Training Done! --- Total training time: {time.time() - start_time} seconds")
    
    
# Entry point for the script
if __name__ == "__main__":
    
    mlflow.set_tracking_uri("https://veeramanicadas-mlflow-server.hf.space")  # Set the MLflow tracking URI
    experiment_name = "fraud_detection_hyperparameter_tuning"  # Define experiment name
    mlflow.set_experiment(experiment_name)
    
    # Define the hyperparameter grids for RandomForest and LogisticRegression
    param_grids = {
        "RandomForest": {
            "classifier__n_estimators": [100, 150],  # Number of estimators in the forest
            "classifier__max_depth": [10, 20, 30],  # Maximum depth of trees
            "classifier__min_samples_split": [2, 5],  # Minimum samples to split a node
            "classifier__min_samples_leaf": [1, 2]  # Minimum samples per leaf
        },
        "LogisticRegression": {
            "classifier__C": [0.1, 1.0, 10],  # Regularization strength
            "classifier__penalty": ['l2'],  # Regularization type
            "classifier__solver": ['lbfgs', 'liblinear']  # Optimization solvers
        }
    }
    
    artifact_path = "fraud_detection_model"  # Path to store the model artifact
    registered_model_name = "fraud_detector_best_model"  # Model registry name

    run_experiment(experiment_name, param_grids, artifact_path, registered_model_name)

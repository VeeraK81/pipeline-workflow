pipeline {
    agent any

    stages {
        stage('Cleanup Existing Containers') {
            steps {
                script {
                    // Stop and remove any existing container with the name "fraud-detection"
                    sh '''
                    if [ $(docker ps -a -q -f name=fraud-detection) ]; then
                        echo "Stopping and removing existing fraud-detection container..."
                        docker stop fraud-detection || true
                        docker rm fraud-detection || true
                        docker rmi fraud-detection-pipeline-image || true
                    fi
                    '''
                }
            }
        }


        stage('Checkout') {
            steps {
                // Checkout the code from the repository
                git branch: 'main', url: 'https://github.com/VeeraK81/pipeline-workflow.git'
            }
        }

        stage('Build Docker Image') {
            steps {
                script {
                    // Build the Docker image using the Dockerfile
                    sh 'docker build -t fraud-detection-pipeline-image .'
                }
            }
        }

        stage('Run Docker Image') {
            steps {
                script {
                    // Run the Docker container in detached mode
                    sh 'docker run -d --name fraud-detection fraud-detection-pipeline-image'

                    // Verify that the container is running
                    def containerStatus = sh(script: "docker inspect -f '{{.State.Running}}' fraud-detection", returnStdout: true).trim()

                    // Check if the container is running
                    if (containerStatus != 'true') {
                        error "The container 'fraud-detection' failed to start."
                    } else {
                        echo "The container 'fraud-detection' is running successfully."
                    }
                }
            }
        }
    



    //     stage('Run Tests Inside Docker Container') {
    //         steps {
    //             withCredentials([
    //                 string(credentialsId: 'mlflow-tracking-uri', variable: 'MLFLOW_TRACKING_URI'),
    //                 string(credentialsId: 'aws-access-key', variable: 'AWS_ACCESS_KEY_ID'),
    //                 string(credentialsId: 'aws-secret-key', variable: 'AWS_SECRET_ACCESS_KEY'),
    //                 string(credentialsId: 'backend-store-uri', variable: 'BACKEND_STORE_URI'),
    //                 string(credentialsId: 'artifact-root', variable: 'ARTIFACT_ROOT')
    //             ]) {
    //                 // Write environment variables to a temporary file
    //                 // KEEP SINGLE QUOTE FOR SECURITY PURPOSES (MORE INFO HERE: https://www.jenkins.io/doc/book/pipeline/jenkinsfile/#handling-credentials)
    //                 script {
    //                     writeFile file: 'env.list', text: '''
    //                     MLFLOW_TRACKING_URI=$MLFLOW_TRACKING_URI
    //                     AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID
    //                     AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY
    //                     BACKEND_STORE_URI=$BACKEND_STORE_URI
    //                     ARTIFACT_ROOT=$ARTIFACT_ROOT
    //                     '''
    //                 }

    //                 // Run a temporary Docker container and pass env variables securely via --env-file
    //                 sh '''
    //                 docker run --rm --env-file env.list \
    //                 final-ml-pipeline-image \
    //                 bash -c "pytest --maxfail=1 --disable-warnings"
    //                 '''
    //             }
    //         }
    //     }
    }

    post {
        always {
            // Clean up workspace and remove dangling Docker images
            sh 'docker system prune -f'
        }
        success {
            echo 'Pipeline completed successfully!'
        }
        failure {
            echo 'Pipeline failed. Check logs for errors.'
        }
    }
}

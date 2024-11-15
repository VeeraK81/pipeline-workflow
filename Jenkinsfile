pipeline {
    agent any

    stages {
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

        // stage('Check AWS Credentials') {
        //     steps {
        //         withCredentials([
        //             string(credentialsId: 'aws-access-key', variable: 'AWS_ACCESS_KEY_ID'),
        //             string(credentialsId: 'aws-secret-key', variable: 'AWS_SECRET_ACCESS_KEY')
        //         ]) {
        //             script {
        //                 sh '''
        //                     if [ -z "$AWS_ACCESS_KEY_ID" ]; then
        //                         echo "AWS_ACCESS_KEY_ID is NOT set!"
        //                     else
        //                         echo "AWS_ACCESS_KEY_ID is set correctly."
        //                         AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID
        //                         AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY
        //                     fi
        //                 '''


        //             }
        //         }
        //     }
        // }




        stage('Run Tests Inside Docker Container') {
            steps {
                withCredentials([
                    string(credentialsId: 'mlflow-tracking-uri', variable: 'MLFLOW_TRACKING_URI'),
                    string(credentialsId: 'aws-access-key', variable: 'AWS_ACCESS_KEY_ID'),
                    string(credentialsId: 'aws-secret-key', variable: 'AWS_SECRET_ACCESS_KEY'),
                    string(credentialsId: 'backend-store-uri', variable: 'BACKEND_STORE_URI'),
                    string(credentialsId: 'artifact-root', variable: 'ARTIFACT_ROOT'),
                    string(credentialsId: 'fraud-detection-bucket-name', variable: 'BUCKET_NAME'),
                    string(credentialsId: 'fraud-detection-file-key', variable: 'FILE_KEY')
                ]) {
                    // Write environment variables to a temporary file
                    // KEEP SINGLE QUOTE FOR SECURITY PURPOSES (MORE INFO HERE: https://www.jenkins.io/doc/book/pipeline/jenkinsfile/#handling-credentials)
                    // script {
                    //     writeFile file: 'env.list', text: '''
                    //     MLFLOW_TRACKING_URI=$MLFLOW_TRACKING_URI
                    //     AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID
                    //     AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY
                    //     BACKEND_STORE_URI=$BACKEND_STORE_URI
                    //     ARTIFACT_ROOT=$ARTIFACT_ROOT
                    //     '''
                    // }

                    // Run a temporary Docker container and pass env variables securely via --env-file
                    // sh '''
                    // docker run --rm --env-file env.list \
                    // fraud-detection-pipeline-image \
                    // bash -c "pytest --maxfail=1 --disable-warnings"
                    // '''

                    script {
                        sh '''
                            docker run \
                                -e AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID \
                                -e AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY \
                                -e MLFLOW_TRACKING_URI=$MLFLOW_TRACKING_URI \
                                -e BACKEND_STORE_URI=$BACKEND_STORE_URI \
                                -e ARTIFACT_ROOT=$ARTIFACT_ROOT \
                                -e BUCKET_NAME=$BUCKET_NAME \
                                -e FILE_KEY=$FILE_KEY \
                                fraud-detection-pipeline-image \
                                bash -c "pytest --maxfail=1 --disable-warnings"
                        '''
                    }
                }
            }
        }
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


        // stage('Run Docker Image') {
        //     steps {
        //         script {
        //             // Run the Docker container in detached mode
        //             sh 'docker run fraud-detection-pipeline-image'

        //             // Verify that the container is running
        //             def containerStatus = sh(script: "docker inspect -f '{{.State.Running}}' fraud-detection", returnStdout: true).trim()

        //             // Check if the container is running
        //             if (containerStatus != 'true') {
        //                 error "The container 'fraud-detection' failed to start."
        //             } else {
        //                 echo "The container 'fraud-detection' is running successfully."
        //             }
        //         }
        //     }
        // }
    

//     post {
//         always {
//             // Clean up workspace and remove dangling Docker images
//             sh 'docker system prune -f'
//         }
//         success {
//             echo 'Pipeline completed successfully!'
//         }
//         failure {
//             echo 'Pipeline failed. Check logs for errors.'
//         }
//     }
// }

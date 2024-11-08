- Machine Learning flow pipeline deployed in Jenkins as pipeline project.

- This container is running inside the Jenkins docker.  To know about this: 

- go to jenkins in docker 
$ docker exec -it <jenkinsContainerId> bash
- navigate to /var/jenkins_home/workspace
$ docker ps
$ docker exec -it <containerId> bash

- to see the files and build details, navigate to /var/jenkins_home/jobs





services:
  mlflow:
    build: .
    image: jedha/sample-mlflow-server
    container_name: mlflow-server
    env_file:
    - .env
    environment:
      - BACKEND_STORE_URI=${BACKEND_STORE_URI}
      - ARTIFACT_ROOT=${ARTIFACT_ROOT} 
      - AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}  # Use environment variable for security
      - AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}  # Use environment variable for security
    ports:
      - ${PORT}  # Expose MLflow on port 8081 mapping 5000
    volumes:
      - mlflow-volume:/var/lib/mlflow/data

volumes:
  mlflow-volume:
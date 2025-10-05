A3: Car Price Prediction (Classification) with MLOps
This repository contains the submission for the AT82.03: Machine Learning course assignment. The project frames the car price prediction problem as a multi-class classification task and implements a full MLOps workflow, including local application deployment via Docker and a complete CI/CD pipeline using GitHub Actions.

Student ID: st126055

Student Name: Supanut Kompayak

📋 Project Overview
The primary goal of this project is to build, evaluate, and deploy a web application that predicts a car's price category. The workflow covers the entire machine learning lifecycle:

Data Preprocessing: A robust Scikit-learn pipeline is used to clean the data, handle missing values, and prepare features for modeling.

Custom Model: The prediction is handled by a custom-built Logistic Regression model, which was trained on the preprocessed data.

Experiment Tracking: All experiments and model versions are tracked using a remote MLflow server.

Application: A web application is built using Dash to provide an interactive user interface for making predictions.

Containerization: The entire application, along with its model and dependencies, is containerized using Docker for consistent and reliable deployment.

CI/CD Automation: A GitHub Actions workflow automates the testing and deployment process to ensure code quality and streamline releases.

🚀 How to Run the Application
The application is containerized with Docker, making it easy to run on any machine.

Prerequisites

Docker Desktop installed and running on your machine.

Git for cloning the repository.

Instructions

Clone the Repository
Open your terminal and clone this repository to your local machine.

git clone [https://github.com/gossbu666/car-prediction-final-st126055.git](https://github.com/gossbu666/car-prediction-final-st126055.git)
cd car-prediction-final-st126055

Build the Docker Image
In the root directory of the project (where the Dockerfile is located), run the following command to build the Docker image. This will package the application, all dependencies, and the trained models.

docker build -t car-prediction-app .

Run the Docker Container
Once the image is built, run it as a container. This command maps port 8050 of the container to port 8050 on your local machine.

docker run -p 8050:8050 car-prediction-app

Access the Application
The application is now running! 🎉 Open your web browser and navigate to:

http://localhost:8050

🤖 CI/CD Automation with GitHub Actions
This project implements a complete Continuous Integration and Continuous Deployment (CI/CD) pipeline using GitHub Actions. The workflow is defined in the .github/workflows/main.yml file and is designed to ensure that every change is automatically tested and deployed.

How It Works

The workflow is automatically triggered on every push to the main branch. It consists of two main jobs that run in sequence:

1. test Job

This is the Continuous Integration (CI) part. Its goal is to verify the integrity and correctness of the code.

It first checks out the code from the repository.

It then sets up a Python environment and installs all the dependencies listed in requirements.txt.

Finally, it runs a series of unit tests located in the /tests directory using pytest. These tests validate critical components, such as ensuring the model's prediction function handles expected inputs and produces outputs of the correct shape.

This job must pass before the deployment job can begin. If any test fails, the workflow stops, preventing faulty code from being deployed.

2. deploy_docker Job

This is the Continuous Deployment (CD) part. It only runs if the test job completes successfully.

This job builds a new Docker image from the Dockerfile using the latest version of the code.

After the image is successfully built, it would typically be pushed to a container registry (like Docker Hub, GitHub Container Registry, etc.). This step ensures that a new, tested version of the application is ready for production.

This automated process guarantees that every change is validated, leading to a more stable and reliable application.


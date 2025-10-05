A3: Car Price Prediction (Classification) with MLOps

Student ID: st126055

Student Name: Supanut Kompayak

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
In the root directory of the project (where the Dockerfile is located), run the following command to build the Docker image. This will package the application, all dependencies, and the trained model artifacts.

docker build -t car-prediction-app .

Run the Docker Container
Once the image is built, run it as a container. This command maps port 8050 of the container to port 8050 on your local machine.

docker run -p 8050:8050 car-prediction-app

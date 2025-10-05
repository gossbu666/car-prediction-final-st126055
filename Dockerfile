# 1. Use an official Python runtime as a parent image
FROM python:3.12-slim

# 2. Set the working directory in the container to /app
WORKDIR /app

# 3. Copy just the requirements file first to leverage Docker cache
COPY requirements.txt .

# 4. Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# --- 5. THE CRUCIAL FIX IS HERE ---
# Copy the CONTENTS of the local 'app' directory into the container's /app directory
COPY app/ .

# 6. Expose the port the app runs on
EXPOSE 8050

# 7. Define the command to run your app (now it's just main.py)
CMD ["python", "app.py"]

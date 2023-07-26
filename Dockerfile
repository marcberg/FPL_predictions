# Use the official Python 3.10 image as a base
FROM python:3.10
# FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the entire project directory into the container at /app
COPY . /app

# Install required Python dependencies
RUN pip install -r requirements.txt

# Set the command to run your ML project
# CMD ["python", "main.py"]
CMD ["python", "main_test_docker.py"]
# Use the official Python image with version 3.10.0 as the base image
FROM python:3.10.0

# Set the working directory in the container
WORKDIR /app

# Copy the requirements.txt file into the container at /app
COPY requirements.txt .

# Install the required packages using pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Run the main.py script
CMD ["python", "main.py"]
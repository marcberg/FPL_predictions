# Use the official Python 3.10 image as a base
FROM python:3.10

# Set the working directory inside the container
WORKDIR /app

# Copy the entire project directory into the container at /app
COPY . /app

# Install required Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set the command to run your ML project
CMD ["python", "main.py"]
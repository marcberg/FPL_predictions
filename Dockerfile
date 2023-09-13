# Use the official Apache Airflow image
FROM apache/airflow:2.7.0



# Set the working directory in the container
WORKDIR /app

# Copy your DAGs and other files into the container
COPY . .

COPY requirements.txt .

# Install required packages from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
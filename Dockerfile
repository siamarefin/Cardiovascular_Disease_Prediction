# Use the official Python image with version 3.8 or later
FROM python:3.11 

# Set the working directory inside the container
WORKDIR /app

# Copy the application files into the container
COPY . /app
COPY files /app/files

# Install system dependencies (if required, e.g., for pandas, numpy, etc.)
RUN apt-get update && apt-get install -y \
    build-essential \
    libssl-dev \
    libffi-dev \
    && apt-get clean

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port that FastAPI will run on
EXPOSE 8000

# Command to run the FastAPI application using Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

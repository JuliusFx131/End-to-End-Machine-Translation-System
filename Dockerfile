# Use a slim Python image
FROM python:3.11.8-slim

# Set the working directory inside the Docker container
WORKDIR /app

# Copy the requirements file to the container
COPY ./serve-requirements.txt .

# Install ctranslate2 and other dependencies
RUN pip install --no-cache-dir ctranslate2==4.3.1 && \
    pip install --no-cache-dir -r serve-requirements.txt && \
    rm -rf /var/lib/apt/lists/*

# Copy the model directory and main script to the container
COPY ./model_dir /app/model_dir
COPY ./main.py /app/main.py

# Set the entrypoint to run the main script
ENTRYPOINT ["python", "-m", "main"]

FROM python:3.10

# Set the working directory
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt /app/

# Install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy the model training script and run it
COPY model_training.py /app/
RUN mkdir -p /app/Models
RUN python model_training.py

# Copy the models directory and server.py files
COPY Models /app/Models
COPY server.py /app/server.py

# Expose the port the app runs on
EXPOSE 5000

# Command to run the Flask app
CMD ["python", "server.py"]
FROM python:3.10

# Set the working directory
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt /app/

# Install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy the application code and data into the container
COPY . .

RUN mkdir -p /app/Models
RUN python model_training.py


# Expose the port the app runs on
EXPOSE 5000

# Command to run the Flask app
CMD ["python", "server.py"]
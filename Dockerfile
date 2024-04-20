# Use an official Python image as a base
FROM python:3.9-slim

# Set the working directory to /violence_detection_project
WORKDIR /app

# Copy the entire violence_detection_project directory
COPY ./violence_detection_project /app/violence_detection_project

# Install the dependencies
RUN pip install -r requirements.txt

# Expose the port
EXPOSE 5000

# Run the command to start the Flask app
CMD ["flask", "run", "--host=0.0.0.0"]





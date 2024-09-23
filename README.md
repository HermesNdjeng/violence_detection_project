# Serve a Violence Detection Model as a Webservice
=====================================================

Serving a violence detection model as a webservice using Flask and Docker.

## Getting Started

Install first all the dependencies of our project by running `pip install -r requirements.txt`

Use `model_definition.ipynb` to train a violence detection model on your dataset and generate a model file (`model.h5`).

Use `appli.py` to wrap the inference logic in a Flask server to serve the model as a REST webservice:

* Execute the command `python appli.py` to run the Flask app.
* Go to the browser and hit the URL `0.0.0.0:5000`.
* Next, run the below command in terminal to query the Flask server to get a reply for the model file provided in this repo:
```
curl -X POST \
  0.0.0.0:5000/ \
  -F "file=@/path/to/video/file.mp4"
```
Replace `/path/to/video/file.mp4` with the actual path to the video file you want to upload and process.

## Building and Running the Docker Image

Run `docker build -t violence-detection-app.` to build the Docker image using the `Dockerfile`. (Pay attention to the period in the `docker build` command)

Run `docker run -p 5000:5000 violence-detection-app` to run the Docker container that got generated using the `violence-detection-app` Docker image. 

Use the below command in terminal to query the Flask server to get a reply for the model file provided in this repo:
```
curl -X POST \
  0.0.0.0:5000/ \
  -F "file=@/path/to/video/file.mp4"
```
Replace `/path/to/video/file.mp4` with the actual path to the video file you want to upload and process.

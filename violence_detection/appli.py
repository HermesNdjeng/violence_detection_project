from flask import Flask, request, jsonify
import numpy as np
import cv2
#import torch
#from torchvision import transforms
import pickle
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
import os
#from tqdm import tqdm
import violence_detection.premodel_core_function as premodel_core_function

app = Flask(__name__)
from tensorflow.keras.models import load_model
model = load_model("model.h5")

if model is None:
    print("Model loading failed. Please check the model file path.")
else:
    print("Model loaded successfully.")

class_names = ['violence', 'non_violence'] # class names

@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        if file:
            # Save the uploaded file to a temporary location
            video_path = os.path.join('', file.filename)
            file.save(video_path)
            video_path_l = [video_path]
            # Process the video to extract features
            total_video = 1 # Assuming you're processing one video at a time
            features = premodel_core_function.frames_extraction(video_path_l, total_video = total_video)
            features = features.reshape((features.shape[0], 16, 2048))
            features = np.expand_dims(features[0], axis=0)

            output = model(features)
            prediction = class_names[1] if output > 0.5 else class_names[0]

            # Clean up the temporary file
            os.remove(video_path)

            return jsonify({'prediction': prediction})
    return render_template('upload.html')

if __name__ == '__main__': 
    app.run(debug=True, host='0.0.0.0', port=5000)
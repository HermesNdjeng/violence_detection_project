import cv2
import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.models import Model
from tensorflow.keras import Input
from tqdm import tqdm
import torch

def preprocess_video(video_path, frame_interval=1, target_size=(224, 224)):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Réduction du taux de trame en fonction de l'intervalle spécifié
        if cap.get(cv2.CAP_PROP_POS_FRAMES) % frame_interval == 0:
            # Redimensionnement du cadre à la taille cible
            frame = cv2.resize(frame, target_size)
            frames.append(frame)
    cap.release()
    return frames

def data_augmentation(frames):
    augmented_frames = []
    for frame in frames:
        # Exemple de transformation: retournement horizontal
        flipped_frame = cv2.flip(frame, 1)
        augmented_frames.append(flipped_frame)
    return augmented_frames

pretrained_model = InceptionV3()
# Create a new model for feature extraction
# Extract features from the second-to-last layer of the InceptionV3 model
pretrained_model = Model(inputs=pretrained_model.input,outputs=pretrained_model.layers[-2].output)
pretrained_model.summary()

def feature_extractor(frame):
    # Expand the dimensions of the frame for model compatibility
    img = np.expand_dims(frame, axis=0)

    # Use the pre-trained feature extraction model to obtain the feature vector
    feature_vector = pretrained_model.predict(img, verbose=0)

    # Return the extracted feature vector
    return feature_vector


def frames_extraction(video_path, SEQUENCE_LENGTH=16, IMAGE_WIDTH=299, IMAGE_HEIGHT=299, total_video=0):
    all_video_features = []

    for pos in tqdm(range(total_video)):
        frames_list = []
        video_reader = cv2.VideoCapture(video_path[pos])
        video_frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
        skip_frames_window = max(int(video_frames_count / SEQUENCE_LENGTH), 1)
        
        features = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH))
        for frame_counter in range(SEQUENCE_LENGTH):
            video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_counter * skip_frames_window)
            success, frame = video_reader.read()
            if not success:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resized_frame = cv2.resize(frame_rgb, (IMAGE_HEIGHT, IMAGE_WIDTH))
            normalized_frame = resized_frame / 255
            features = feature_extractor(normalized_frame) # Assuming feature_extractor is defined elsewhere
            frames_list.append(features)

        # Pad the frames_list with zeros if it's shorter than SEQUENCE_LENGTH
        while len(frames_list) < SEQUENCE_LENGTH:
            frames_list.append(np.zeros_like(features)) # Assuming features is a NumPy array

        all_video_features.append(frames_list)
        video_reader.release()

    # Convert the list of features to a numpy array
    return np.array(all_video_features)




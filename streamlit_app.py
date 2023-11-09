import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import joblib  # You can also use 'pickle' for older Scikit-Learn models
from sklearn import svm
import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

# Function to extract frames from a video
def extract_frames(video_path, output_directory, frame_interval=2):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    # Create the output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)

    frame_count = 0

    while True:
        # Read a frame from the video
        ret, frame = cap.read()

        if not ret:
            break

# Skip frames based on the desired frame rate (fps)
        for _ in range(fps - 1):
                cap.read()

        frame_count += 1

        if frame_count % frame_interval == 0:
                # Save the frame as an image
                frame_filename = os.path.join(output_directory, f"frame_{frame_count:04d}.jpg")
                cv2.imwrite(frame_filename, frame)
    # Release the video file
    cap.release()

# Specify the path to the video and the output directory
video_path = '/content/4a311ee8d57e190184eb2c1844262ea9.mp4'
output_directory = '/content/prediction7'

# Call the function to extract frames
extract_frames(video_path, output_directory)


# Load the pre-trained ResNet50 model
model = ResNet50(weights='imagenet', include_top=False)
feature_list = []

# Define the folder containing your video files
video_folder = '/content/prediction'

# Loop through each video file in the folder
for video_filename in os.listdir(video_folder):
    video_path = os.path.join(video_folder, video_filename)

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Initialize a list to store frame-level features
    video_features = []

    # Read frames from the video
    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # Preprocess the frame and extract features
        frame = cv2.resize(frame, (224, 224))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = img_to_array(frame)
        frame = preprocess_input(frame)
        frame = tf.expand_dims(frame, axis=0)

        features = model.predict(frame)
        video_features.append(features.flatten())

    # Concatenate or stack frame features horizontally to create a feature matrix for the video
    video_feature_matrix = np.vstack(video_features)  # Stack the frame features vertically

    # Aggregate the frame features to obtain a feature vector for the video
    video_feature_vector = np.mean(video_feature_matrix, axis=0)  # You can use mean, max, or other aggregation functions

    # Append the video feature vector to the list of video features
    feature_list.append(video_feature_vector)

    # Release the video capture
    cap.release()

# Save the extracted features to a file (e.g., in NumPy format)
features_array = np.array(feature_list)
np.save('pred7_features.npy', features_array)


# Replace 'your_model_file.pkl' with the actual filename of your saved model.
loaded_model = joblib.load('svm_model.pkl')

# Make predictions on new data
new_data = np.load('/content/pred3_features.npy')
new_data = new_data.reshape(new_data.shape[0], -1)
predictions = loaded_model.predict(new_data)
print(predictions)

# Calculate the ratio of 1s in the prediction
ratio_of_ones = sum(1 for pred in predictions if pred == 1) / len(predictions)

# Set a threshold for the ratio to determine "high amount"
threshold = 0.7  # Adjust as needed

if ratio_of_ones > threshold:
    print("It is a real video")
else:
    print("It is not a real video")




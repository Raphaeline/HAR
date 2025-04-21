import os

import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


from TNet import TNet

def rotate_z(points, angle):
    cos_theta = np.cos(angle)
    sin_theta = np.sin(angle)
    rotation_matrix = np.array([
        [cos_theta, -sin_theta, 0],
        [sin_theta, cos_theta, 0],
        [0, 0, 1]
    ])
    rotated_points = np.dot(points, rotation_matrix.T)
    return rotated_points

def farthestPointSampling(points, numSamples, seed):
    np.random.seed(seed)
    numPoints = points.shape[0]
    selectedIndices = np.zeros(numSamples, dtype=int)
    initialIndex = np.random.randint(numPoints)
    selectedIndices[0] = initialIndex
    distances = np.linalg.norm(points - points[initialIndex, :], axis=1)
    for i in range(1, numSamples):
        lastSelected = selectedIndices[i - 1]
        newDistances = np.linalg.norm(points - points[lastSelected, :], axis=1)
        distances = np.minimum(distances, newDistances)
        selectedIndex = np.argmax(distances)
        selectedIndices[i] = selectedIndex
    return selectedIndices

def processAndPredict():
    numSamples = 256
    inputFolder = "./processingData/Garry"
    scriptDir = os.path.dirname(os.path.abspath(__file__))
    modelPath = os.path.join(scriptDir, "HARmodelPNet2.h5")
    model = tf.keras.models.load_model(modelPath, custom_objects={"TNet": TNet})
    
    class_mapping = {0: "Stand", 1: "Sit", 2: "Walk", 3: "Fall"}
    
    for root, dirs, files in os.walk(inputFolder):
        for file in files:
            if file.endswith(".csv"):
                filePath = os.path.join(root, file)
                print(f"\nProcessing file: {file}")
                
                df = pd.read_csv(filePath)
                if 'x' in df.columns and 'y' in df.columns and 'z' in df.columns:
                    points = df[['x', 'y', 'z']].to_numpy()
                else:
                    points = df.to_numpy()

                # Store predictions for all seeds
                all_predictions = []
                all_confidences = []
                all_seeds = []
                
                for seed in range(6):
                    indices = farthestPointSampling(points, numSamples, seed)
                    sampled_points = points[indices]
                    
                    if seed > 0:
                        angle = seed * 60 * (np.pi / 180) 
                        sampled_points = rotate_z(sampled_points, angle)
                    
                    frames_adjusted = pad_sequences(
                        [sampled_points], 
                        maxlen=256,
                        dtype='float32',
                        padding='post', 
                        truncating='post' 
                    )[0]
                    
                    X = np.expand_dims(frames_adjusted, axis=0)
                    
                    predictions = model.predict(X)
                    predicted_classes = np.argmax(predictions, axis=1)
                    confidence = np.max(predictions[0])
                    
                    # Skip predictions with confidence > 0.9999
                    if confidence >= 0.9:
                        continue
                    
                    predicted_class_name = class_mapping.get(predicted_classes[0], "Unknown")
                    
                    # Check if the predicted class is valid
                    if predicted_class_name not in ["Stand", "Sit", "Walk", "Fall"]:
                        predicted_class_name = "Unknown"
                        print(f"\nSeed {seed+1}: unknown (Invalid prediction) {confidence:.4f}")
                        continue
                    
                    all_predictions.append(predicted_class_name)
                    all_confidences.append(confidence)
                    all_seeds.append(seed)
                    
                    print(f"\nSeed {seed+1}: {predicted_class_name} {confidence:.4f}")
                    print(f"All confidence scores: {predictions[0]}")
                
                # Find the prediction with highest confidence
                if all_confidences:
                    max_confidence_idx = np.argmax(all_confidences)
                    best_seed = all_seeds[max_confidence_idx]
                    best_prediction = all_predictions[max_confidence_idx]
                    best_confidence = all_confidences[max_confidence_idx]
                    
                    print(f"Best prediction from Seed {best_seed+1}:")
                    print(f"Class: {best_prediction}")
                    print(f"Confidence: {best_confidence:.4f}")
                else:
                    print("\nNo valid predictions found for any seed.")

if __name__ == '__main__':
    processAndPredict()
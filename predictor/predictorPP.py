import os

import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras.preprocessing.sequence import pad_sequences


class TNet(layers.Layer):
    def __init__(self, k, **kwargs):  # Tambahkan **kwargs untuk menangani argumen tambahan
        super(TNet, self).__init__(**kwargs)  # Pastikan super() menangani kwargs
        self.k = k
        self.conv1 = layers.Conv1D(64, 1, activation='relu')
        self.conv2 = layers.Conv1D(128, 1, activation='relu')
        self.conv3 = layers.Conv1D(1024, 1, activation='relu')
        self.fc1 = layers.Dense(512, activation='relu')
        self.fc2 = layers.Dense(256, activation='relu')
        self.fc3 = layers.Dense(k * k, activation=None, kernel_initializer='glorot_uniform')
        self.batch_norm1 = layers.BatchNormalization()
        self.batch_norm2 = layers.BatchNormalization()
        self.batch_norm3 = layers.BatchNormalization()


    def build(self, input_shape):
        self.built = True

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.batch_norm1(x)
        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = self.conv3(x)
        x = self.batch_norm3(x)
        x = layers.GlobalMaxPooling1D()(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        x = tf.reshape(x, (-1, self.k, self.k))
        identity = tf.eye(self.k, batch_shape=[tf.shape(inputs)[0]])
        x = x + identity

        # x_transpose = tf.transpose(x, perm=[0, 2, 1])
        # product = tf.matmul(x, x_transpose)
        # orth_loss = self.reg_factor * tf.reduce_mean(tf.square(product - identity))
        # self.add_loss(orth_loss)

        transformed = tf.matmul(inputs, x) 
        return transformed

def setAbstractionLayer(x, num_samples, k, mlp_channels):

    centroids = layers.Lambda(lambda x: x[:, :num_samples, :])(x)
    grouped = layers.Lambda(lambda c: tf.tile(tf.expand_dims(c, axis=2), [1, 1, k, 1]))(centroids)

    for channels in mlp_channels:
        grouped = layers.Conv2D(channels, kernel_size=(1,1), activation='relu')(grouped)
        grouped = layers.BatchNormalization()(grouped)

    new_features = layers.Lambda(lambda x: tf.reduce_max(x, axis=2))(grouped)
    return centroids, new_features


customObjects = {"TNet": TNet}
get_custom_objects().update(customObjects)

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
    customObjects = custom_objects={"TNet": TNet}
    modelPath = os.path.join(scriptDir, "HARmodelPNetPP.h5")
    model = tf.keras.models.load_model(modelPath, customObjects)
    

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
                    
                    if confidence >= 1:
                        continue
                    
                    predicted_class_name = class_mapping.get(predicted_classes[0], "Unknown")
                    print(f"\nSeed {seed+1}:")
                    print(f"Class: {predicted_class_name}")
                    print(f"Confidence: {confidence:.4f}")
                    print(f"All confidence scores: {predictions[0]}")

if __name__ == '__main__':
    processAndPredict()
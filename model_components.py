import tensorflow as tf
from tensorflow.keras import layers

class TNet(layers.Layer):
    def __init__(self, k, **kwargs):
        super(TNet, self).__init__(**kwargs)
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

        transformed = tf.matmul(inputs, x) 
        return transformed

def setAbstractionLayer(x, num_samples, k, mlp_channels):
    # Define output shapes for Lambda layers
    def get_centroids_shape(input_shape):
        return (input_shape[0], num_samples, input_shape[2])
        
    def get_grouped_shape(input_shape):
        return (input_shape[0], input_shape[1], k, input_shape[2])
        
    def get_features_shape(input_shape):
        return (input_shape[0], input_shape[1], input_shape[3])

    # Create centroids and grouped points with specified output shapes
    centroids = layers.Lambda(
        lambda x: x[:, :num_samples, :],
        output_shape=get_centroids_shape
    )(x)
    
    grouped = layers.Lambda(
        lambda c: tf.tile(tf.expand_dims(c, axis=2), [1, 1, k, 1]),
        output_shape=get_grouped_shape
    )(centroids)

    # Apply convolutional layers
    for channels in mlp_channels:
        grouped = layers.Conv2D(channels, kernel_size=(1,1), activation='relu')(grouped)
        grouped = layers.BatchNormalization()(grouped)

    # Get new features with specified output shape
    new_features = layers.Lambda(
        lambda x: tf.reduce_max(x, axis=2),
        output_shape=get_features_shape
    )(grouped)
    
    return centroids, new_features 
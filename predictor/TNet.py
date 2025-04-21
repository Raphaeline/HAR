import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import get_custom_objects

class TNet(layers.Layer):
    def __init__(self, k, reg_factor=0.01, **kwargs):  # Tambahkan **kwargs untuk menangani argumen tambahan
        super(TNet, self).__init__(**kwargs)  # Pastikan super() menangani kwargs
        self.k = k
        self.reg_factor = reg_factor
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

        x_transpose = tf.transpose(x, perm=[0, 2, 1])
        product = tf.matmul(x, x_transpose, transpose_b=True)
        orth_loss = self.reg_factor * tf.reduce_mean(tf.square(product - identity))
        self.add_loss(orth_loss)

        transformed = tf.matmul(inputs, x) 
        return transformed



customObjects = {"TNet": TNet}
get_custom_objects().update(customObjects)
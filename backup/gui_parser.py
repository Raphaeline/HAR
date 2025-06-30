"# ----- Imports -------------------------------------------------------"

from PyQt5.QtCore import QThread, pyqtSignal
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras import layers

from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import tensorflow as tf
from sklearn.mixture import GaussianMixture
from PyQt5.QtCore import QObject, pyqtSignal
from PyQt5.QtCore import QThread
from parseFrame import *
from queue import Queue
import threading
import datetime
import math
import numpy as np
import time
import serial
import struct
import sys
import os
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Standard Imports

# Local Imports


class uartParser(QObject):
    predictionResult = pyqtSignal(str, float)

    def __init__(self, type='SDK Out of Box Demo'):
        super().__init__()
        # Set this option to 1 to save UART output from the radar device
        self.currentThread = None
        self.replay = 0
        self.uartCounter = 0
        self.first_file = True
        self.filepath = datetime.datetime.now().strftime("%m_%d_%Y_%H_%M_%S")

        # Optimized windowing system with stride 10
        self.window_size = 30  # Number of timestamps in window
        self.window_stride = 10  # Stride for windowing
        self._stride_counter = 0  # Internal counter for stride
        self.timestamp_data = {}  # Store raw data for each timestamp
        self.timestamp_order = []  # Keep track of timestamp order
        self.last_prediction_time = 0
        self.prediction_interval = 0.1  # Predict every 100ms
        self.points_count = 0  # Track total points for efficiency
        self.prediction_ready = False  # Flag to avoid redundant checks
        self.POINTS_PER_TIMESTAMP = 150  # Target points per timestamp for GMM

        # Tambahkan set untuk tracking unique timestamps
        self.unique_timestamps = set()

        if (type == DEMO_NAME_OOB):
            self.parserType = "DoubleCOMPort"
        elif (type == DEMO_NAME_LRPD):
            self.parserType = "DoubleCOMPort"
        elif (type == DEMO_NAME_3DPC):
            self.parserType = "DoubleCOMPort"
        elif (type == DEMO_NAME_SOD):
            self.parserType = "DoubleCOMPort"
        elif (type == DEMO_NAME_VITALS):
            self.parserType = "DoubleCOMPort"
        elif (type == DEMO_NAME_MT):
            self.parserType = "DoubleCOMPort"
        elif (type == DEMO_NAME_GESTURE):
            self.parserType = "DoubleCOMPort"
        elif (type == DEMO_NAME_x432_OOB):
            self.parserType = "SingleCOMPort"
        elif (type == DEMO_NAME_x432_GESTURE):
            self.parserType = "SingleCOMPort"
        # TODO Implement these
        elif (type == "Replay"):
            self.replay = 1
        else:
            print("ERROR, unsupported demo type selected!")

        # Data storage
        self.now_time = datetime.datetime.now().strftime('%Y%m%d-%H%M')

    def disconnectComPort(self):
        try:
            if self.currentThread is not None:
                if self.currentThread.isRunning():
                    self.currentThread.quit()
            if hasattr(self, 'cliPort') and self.cliPort is not None:
                self.cliPort.close()
                self.cliPort = None
            if hasattr(self, 'dataPort') and self.dataPort is not None:
                self.dataPort.close()
                self.dataPort = None
            print("Serial ports disconnected.")
        except Exception as e:
            print("Error disconnecting COM ports:", e)

    def cleanup_old_data(self):
        """Clean up old timestamp data to prevent memory accumulation - optimized"""
        current_time = time.time()
        cutoff_time = current_time - 10.0

        # Convert string timestamps to datetime for comparison - optimized
        timestamps_to_remove = []
        for ts_str in self.timestamp_order:
            try:
                ts_dt = datetime.datetime.strptime(
                    ts_str, "%Y-%m-%d %H:%M:%S.%f")
                ts_timestamp = ts_dt.timestamp()
                if ts_timestamp < cutoff_time:
                    timestamps_to_remove.append(ts_str)
            except:
                continue

        # Remove old timestamps - optimized
        if timestamps_to_remove:
            for ts in timestamps_to_remove:
                if ts in self.timestamp_data:
                    removed_points = len(self.timestamp_data[ts])
                    self.points_count -= removed_points
                    del self.timestamp_data[ts]
                if ts in self.timestamp_order:
                    self.timestamp_order.remove(ts)

            print(
                f"Cleaned up {len(timestamps_to_remove)} old timestamps, {self.points_count} points remaining")

    # This function is always called - first read the UART, then call a function to parse the specific demo output
    # This will return 1 frame of data. This must be called for each frame of data that is expected. It will return a dict containing all output info
    # Point Cloud and Target structure are liable to change based on the lab. Output is always cartesian.
    # DoubleCOMPort means this function refers to the xWRx843 family of devices.

    def _prepare_points(self):
        """
        Prepare points by:
        1. Grouping points by timestamp
        2. Taking the last 30 timestamps
        3. Sampling 150 points from each timestamp
        Returns array with shape (4500, 5) including SNR
        """
        # Group points by timestamp - optimized
        timestamps = self.batch_data[:, 0]  # Get all timestamps
        unique_timestamps = np.unique(timestamps)

        # If we don't have enough timestamps, return None
        if len(unique_timestamps) < self.MIN_TIMESTAMPS_REQUIRED:
            print(
                f"Not enough timestamps: {len(unique_timestamps)}/{self.MIN_TIMESTAMPS_REQUIRED}")
            # Emit last prediction while collecting
            self.predictionResult.emit(*self._last_prediction)
            return None

        # Process each timestamp in order - optimized
        processed_frames = []
        for ts in unique_timestamps:
            # Get points for this timestamp
            mask = timestamps == ts
            frame_points = self.batch_data[mask, 1:5].astype(
                np.float32)  # Only 4 features

            # Skip normalization - use raw values directly
            # More efficient sampling
            if len(frame_points) >= self.POINTS_PER_TIMESTAMP:
                # Use systematic sampling for better performance
                step = len(frame_points) // self.POINTS_PER_TIMESTAMP
                indices = np.arange(0, len(frame_points), step)[
                    :self.POINTS_PER_TIMESTAMP]
            else:
                # Efficient oversampling
                indices = np.random.choice(
                    len(frame_points), self.POINTS_PER_TIMESTAMP, replace=True)

            sampled_points = frame_points[indices]
            processed_frames.append(sampled_points)

        # Stack frames to maintain 3D shape: (num_frames, num_points, num_features)
        stacked_frames = np.stack(processed_frames, axis=0)
        print(
            f"Processed {len(unique_timestamps)} timestamps with shape: {stacked_frames.shape}")
        return stacked_frames

    def readAndParseUartDoubleCOMPort(self):
        """
        Read and parse the UART data, then save the point cloud to a CSV file.
        """
        self.fail = 0
        outputDict = {}  # Initialize outputDict before using it.

        if (self.replay):
            return self.replayHist()

        # Find magic word, and therefore the start of the frame
        index = 0
        magicByte = self.dataCom.read(1)
        frameData = bytearray(b'')

        while (1):
            # If the device doesn't transmit any data, the COMPort read function will eventually timeout
            if (len(magicByte) < 1):
                print("ERROR: No data detected on COM Port, read timed out")
                magicByte = self.dataCom.read(1)

            # Found matching byte
            elif (magicByte[0] == UART_MAGIC_WORD[index]):
                index += 1
                frameData.append(magicByte[0])
                if (index == 8):  # Found the full magic word
                    break
                magicByte = self.dataCom.read(1)
            else:
                if (index == 0):
                    magicByte = self.dataCom.read(1)
                index = 0  # Reset index
                frameData = bytearray(b'')  # Reset current frame data

        # Read in version from the header
        versionBytes = self.dataCom.read(4)
        frameData += bytearray(versionBytes)

        # Read in length from header
        lengthBytes = self.dataCom.read(4)
        frameData += bytearray(lengthBytes)
        frameLength = int.from_bytes(lengthBytes, byteorder='little')

        # Subtract bytes that have already been read, IE magic word, version, and length
        frameLength -= 16

        # Read in the rest of the frame
        frameData += bytearray(self.dataCom.read(frameLength))

        # Make sure outputDict is assigned properly.
        if (self.parserType == "DoubleCOMPort"):
            outputDict = parseStandardFrame(frameData)
        else:
            print('FAILURE: Bad parserType')

        # Ensure point cloud data exists and convert if necessary
        pointCloudData = outputDict.get('pointCloud', [])
        if isinstance(pointCloudData, np.ndarray) and pointCloudData.size > 0:
            # Convert to list if it's a numpy array
            pointCloudData = pointCloudData.tolist()

        # Tambahkan timestamp ke set
        frameTimestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
        self.unique_timestamps.add(frameTimestamp)

        # Add data to windowing system
        if pointCloudData and isinstance(pointCloudData, list) and len(pointCloudData) > 0:
            # Store points for this timestamp - optimized
            timestamp_points = []
            valid_points = 0

            for point in pointCloudData:
                if len(point) >= 5:  # Ensure we have all 4 features (x, y, z, doppler)
                    try:
                        features = [float(point[i])
                                    for i in range(5)]  # Only 4 features
                        timestamp_points.append(features)
                        valid_points += 1
                    except (ValueError, TypeError):
                        continue  # Skip invalid points silently

            if valid_points > 0:  # Only add if we have valid points
                self.timestamp_data[frameTimestamp] = timestamp_points
                self.timestamp_order.append(frameTimestamp)
                self.points_count += valid_points

                # Maintain window size by removing oldest timestamps
                while len(self.timestamp_order) > self.window_size:
                    oldest_timestamp = self.timestamp_order.pop(0)
                    if oldest_timestamp in self.timestamp_data:
                        removed_points = len(
                            self.timestamp_data[oldest_timestamp])
                        self.points_count -= removed_points
                        del self.timestamp_data[oldest_timestamp]

                # Set prediction ready flag with stride
                if len(self.timestamp_order) >= self.window_size:
                    self._stride_counter += 1
                    if self._stride_counter % self.window_stride == 0:
                        self.prediction_ready = True
                    else:
                        self.prediction_ready = False

                # Debug: Show acquisition progress
                print(
                    f"[Flow 1] Acquisition: Timestamp {frameTimestamp} - {valid_points} points collected")
                print(
                    f"[Flow 1] Total timestamps: {len(self.timestamp_order)}/{self.window_size}")

        # Check if we should make a prediction (stride 1) - optimized check
        currentTime = time.time()
        if (self.prediction_ready and
                currentTime - self.last_prediction_time >= self.prediction_interval):

            try:
                # Prepare window data using GMM completion
                window_data = self.prepare_window_data()

                if window_data is not None:
                    # Buat thread predictor baru dengan data yang sudah di-GMM
                    self.predThread = HARPredictorGMM(window_data)
                    # Hubungkan signal predictionResult ke handler
                    self.predThread.predictionResult.connect(
                        self.handlePrediction)
                    # Mulai thread
                    self.predThread.start()
                    print(
                        f"[uartParser] Started GMM prediction thread with window shape: {window_data.shape}")
                    self.last_prediction_time = currentTime
                    self.prediction_ready = False  # Reset flag
                else:
                    print(
                        f"[uartParser] Not enough timestamps for windowing: {len(self.timestamp_order)}/{self.window_size}")

            except Exception as e:
                print(f"Error starting prediction thread: {e}")

        # Clean up old data periodically - less frequent
        if len(self.timestamp_order) > self.window_size * 3:
            self.cleanup_old_data()

        return outputDict

    def getRecordedBatches(self):
        return self.recordedBatches

    def handlePrediction(self, label, confidence, predictionList):
        print(
            f"[handlePrediction] Label: {label}, Confidence: {confidence:.2f}%")
        try:
            self.predictionResult.emit(str(label), float(confidence))
            print(
                f"[uartParser] Emitted prediction: {label}, {confidence:.2f}%")
        except Exception as e:
            print(f"Error emitting prediction result: {e}")

    # This function is identical to the readAndParseUartDoubleCOMPort function, but it's modified to work for SingleCOMPort devices in the xWRLx432 family

    def readAndParseUartSingleCOMPort(self):
        # Reopen CLI port
        if (self.cliCom.isOpen() == False):
            print("Reopening Port")
            self.cliCom.open()

        self.fail = 0
        if (self.replay):
            return self.replayHist()

        # Find magic word, and therefore the start of the frame
        index = 0
        magicByte = self.cliCom.read(1)
        frameData = bytearray(b'')
        while (1):
            # If the device doesnt transmit any data, the COMPort read function will eventually timeout
            # Which means magicByte will hold no data, and the call to magicByte[0] will produce an error
            # This check ensures we can give a meaningful error
            if (len(magicByte) < 1):
                print("ERROR: No data detected on COM Port, read timed out")
                print(
                    "\tBe sure that the device is in the proper mode, and that the cfg you are sending is valid")
                magicByte = self.cliCom.read(1)

            # Found matching byte
            elif (magicByte[0] == UART_MAGIC_WORD[index]):
                index += 1
                frameData.append(magicByte[0])
                if (index == 8):  # Found the full magic word
                    break
                magicByte = self.cliCom.read(1)

            else:
                # When you fail, you need to compare your byte against that byte (ie the 4th) AS WELL AS compare it to the first byte of sequence
                # Therefore, we should only read a new byte if we are sure the current byte does not match the 1st byte of the magic word sequence
                if (index == 0):
                    magicByte = self.cliCom.read(1)
                index = 0  # Reset index
                frameData = bytearray(b'')  # Reset current frame data

        # Read in version from the header
        versionBytes = self.cliCom.read(4)

        frameData += bytearray(versionBytes)

        # Read in length from header
        lengthBytes = self.cliCom.read(4)
        frameData += bytearray(lengthBytes)
        frameLength = int.from_bytes(lengthBytes, byteorder='little')

        # Subtract bytes that have already been read, IE magic word, version, and length
        # This ensures that we only read the part of the frame in that we are lacking
        frameLength -= 16

        # Read in rest of the frame
        frameData += bytearray(self.cliCom.read(frameLength))

        # frameData now contains an entire frame, send it to parser
        if (self.parserType == "SingleCOMPort"):
            outputDict = parseStandardFrame(frameData)
        else:
            print('FAILURE: Bad parserType')

        return outputDict

    def connectComPorts(self, cliCom, dataCom):
        self.cliCom = serial.Serial(
            cliCom, 115200, parity=serial.PARITY_NONE, stopbits=serial.STOPBITS_ONE, timeout=0.6)
        self.dataCom = serial.Serial(
            dataCom, 921600, parity=serial.PARITY_NONE, stopbits=serial.STOPBITS_ONE, timeout=0.6)
        self.dataCom.reset_output_buffer()
        print('Connected')

    # Separate connectComPort (not PortS) for IWRL6432 because it only uses one port
    def connectComPort(self, cliCom, cliBaud=115200):
        # Longer timeout time for IWRL6432 to support applications with low power / low update rate
        self.cliCom = serial.Serial(
            cliCom, cliBaud, parity=serial.PARITY_NONE, stopbits=serial.STOPBITS_ONE, timeout=4)
        self.cliCom.reset_output_buffer()
        print('Connected (one port)')

    # send cfg over uart
    def sendCfg(self, cfg):
        # Ensure each line ends in \n for proper parsing
        for i, line in enumerate(cfg):
            # Remove empty lines from cfg
            if (line == '\n'):
                cfg.remove(line)
            # add a newline to the end of every line (protects against last line not having a newline at the end of it)
            elif (line[-1] != '\n'):
                cfg[i] = cfg[i] + '\n'

        for line in cfg:
            time.sleep(.03)  # Line delay

            if (self.cliCom.baudrate == 1250000):
                for char in [*line]:
                    # Character delay. Required for demos which are 1250000 baud by default else characters are skipped
                    time.sleep(.001)
                    self.cliCom.write(char.encode())
            else:
                self.cliCom.write(line.encode())

            ack = self.cliCom.readline()
            print(ack)
            ack = self.cliCom.readline()
            print(ack)
            splitLine = line.split()
            # The baudrate CLI line changes the CLI baud rate on the next cfg line to enable greater data streaming off the IWRL device.
            if (splitLine[0] == "baudRate"):
                try:
                    self.cliCom.baudrate = int(splitLine[1])
                except:
                    print("Error - Invalid baud rate")
                    sys.exit(1)
        # Give a short amount of time for the buffer to clear
        time.sleep(0.03)
        self.cliCom.reset_input_buffer()
        # NOTE - Do NOT close the CLI port because 6432 will use it after configuration

    def sendLine(self, line):
        self.cliCom.write(line.encode())
        ack = self.cliCom.readline()
        print(ack)
        ack = self.cliCom.readline()
        print(ack)

    def gmm_complete_timestamp(self, points, target_count=150, n_components=5):
        """
        Use GMM to complete a timestamp to have exactly target_count points.
        If points < target_count: use GMM to generate additional points
        If points > target_count: sample target_count points
        """
        if len(points) == 0:
            return np.zeros((target_count, 5))

        points_array = np.array(points, dtype=np.float32)

        if len(points_array) >= target_count:
            # Sample target_count points if we have more
            indices = np.random.choice(
                len(points_array), target_count, replace=False)
            result = points_array[indices]
            print(
                f"[Flow 2] GMM: Sampled {len(points_array)} → {target_count} points")
            return result

        elif len(points_array) < n_components:
            # If too few points, just repeat the available points
            repeat_count = target_count // len(points_array) + 1
            repeated = np.tile(points_array, (repeat_count, 1))
            result = repeated[:target_count]
            print(
                f"[Flow 2] GMM: Repeated {len(points_array)} → {target_count} points")
            return result

        else:
            # Use GMM to generate additional points
            try:
                gmm = GaussianMixture(n_components=min(n_components, len(points_array)),
                                      covariance_type='full', random_state=42)
                gmm.fit(points_array)

                # Generate additional points needed
                additional_needed = target_count - len(points_array)
                generated_points = gmm.sample(n_samples=additional_needed)[0]

                completed_points = np.vstack([points_array, generated_points])
                print(
                    f"[Flow 2] GMM: Generated {len(points_array)} → {target_count} points using GMM")
                return completed_points.astype(np.float32)

            except Exception as e:
                print(f"GMM completion failed: {e}, using simple padding")
                # Fallback: repeat points
                repeat_count = target_count // len(points_array) + 1
                repeated = np.tile(points_array, (repeat_count, 1))
                result = repeated[:target_count]
                print(
                    f"[Flow 2] GMM: Fallback padding {len(points_array)} → {target_count} points")
                return result.astype(np.float32)

    def prepare_window_data(self):

        if len(self.timestamp_order) < self.window_size:
            return None
        recent_timestamps = self.timestamp_order[-self.window_size:]
        completed_frames = []

        print(
            f"[Flow 3] Windowing: Processing {len(recent_timestamps)} timestamps")

        for i, ts in enumerate(recent_timestamps):
            if ts in self.timestamp_data:
                raw_points = self.timestamp_data[ts]

                completed_points = self.gmm_complete_timestamp(
                    raw_points, self.POINTS_PER_TIMESTAMP)
                completed_frames.append(completed_points)
                print(
                    f"[Flow 3] Windowing: Timestamp {i+1}/30 - {len(raw_points)} → {len(completed_points)} points")
            else:
                # If timestamp missing, create empty frame
                empty_frame = np.zeros((self.POINTS_PER_TIMESTAMP, 5), dtype=np.float32)
                completed_frames.append(empty_frame)
                print(
                    f"[Flow 3] Windowing: Timestamp {i+1}/30 - Empty frame (150 points)")

        # Stack all frames: (30 timestamps, 150 points, 4 features)
        window_data = np.stack(completed_frames, axis=0)
        print(f"[Flow 3] Windowing: Final window shape: {window_data.shape}")
        return window_data


class TNet(layers.Layer):
    # Tambahkan **kwargs untuk menangani argumen tambahan
    def __init__(self, k, reg_factor=0.001, **kwargs):
        # Pastikan super() menangani kwargs
        super(TNet, self).__init__(**kwargs)
        self.k = k
        self.reg_factor = reg_factor
        self.conv1 = layers.Conv1D(64, 1, activation='relu')
        self.conv2 = layers.Conv1D(128, 1, activation='relu')
        self.conv3 = layers.Conv1D(1024, 1, activation='relu')
        self.fc1 = layers.Dense(512, activation='relu')
        self.fc2 = layers.Dense(256, activation='relu')
        self.fc3 = layers.Dense(k * k, activation=None,
                                kernel_initializer='glorot_uniform')
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
        orth_loss = self.reg_factor * \
            tf.reduce_mean(tf.square(product - identity))
        self.add_loss(orth_loss)

        transformed = tf.matmul(inputs, x)
        return transformed


get_custom_objects().update({"TNet": TNet})


def GMM(referenceDf, partialPoints, targetCount=150, nComponents=5):
    gmm = GaussianMixture(n_components=nComponents,
                          covariance_type='full', random_state=42)
    gmm.fit(referenceDf[['x', 'y', 'z', 'doppler', 'snr']].values)
    toSample = targetCount - len(partialPoints)
    sampledPoints = gmm.sample(n_samples=toSample)[0]
    completed = np.vstack([partialPoints, sampledPoints])
    return completed


def bootstrap(df, targetCount=150, nComponents=5, minPartial=8):
    groupSizes = df.groupby('timestamp').size()
    referenceTimestamp = groupSizes.idxmax()
    referenceDf = df[df['timestamp'] ==
                     referenceTimestamp][['x', 'y', 'z', 'doppler', 'snr']]

    processedList = []
    for timestamp, group in df.groupby('timestamp'):
        if len(group) >= targetCount or len(group) < max(minPartial, nComponents):
            continue
        partialPoints = group[['x', 'y', 'z', 'doppler', 'snr']].values
        completedPoints = GMM(
            referenceDf, partialPoints, targetCount, nComponents)
        completedDf = pd.DataFrame(completedPoints, columns=[
                                   'x', 'y', 'z', 'doppler', 'snr'])
        completedDf['timestamp'] = timestamp
        completedDf['numFrame'] = np.arange(1, targetCount + 1)
        processedList.append(completedDf)

    return pd.concat(processedList, ignore_index=True) if processedList else None


def getBit(byte, bitNum):
    mask = 1 << bitNum
    if (byte & mask):
        return 1
    else:
        return 0


class HARPredictorGMM(QThread):
    predictionResult = pyqtSignal(str, float, list)
    _loadedModel = None
    _last_prediction = ("Stand", 100.0, [1.0, 0.0, 0.0, 0.0])  # Default values

    def __init__(self, window_data, modelPath=None, parent=None):
        super().__init__(parent)
        self.window_data = window_data.astype(np.float32)
        self.modelPath = modelPath or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "Bootstrap-Balanced.h5"))

    def run(self):
        if HARPredictorGMM._loadedModel is None:
            print("🔁 Loading GMM model for the first time")
            try:
                HARPredictorGMM._loadedModel = load_model(self.modelPath, custom_objects={"TNet": TNet}, compile=False)
                print("GMM model loaded successfully")
            except Exception as e:
                print(f"Error loading GMM model: {e}")
                self.predictionResult.emit("Error", 0.0, [0.0, 0.0, 0.0, 0.0])
                return

        model = HARPredictorGMM._loadedModel

        try:

            print(
                f"[GMM Prediction] Input shape: {self.window_data.shape}")

            total_points = self.window_data.shape[0] * self.window_data.shape[1]
            reshaped_points = self.window_data.reshape(1, total_points, 5)
            print(f"[GMM Prediction] Reshaped to: {reshaped_points.shape}")

            prediction = model.predict(reshaped_points, verbose=0)

            labelIdx = np.argmax(prediction)
            # Convert to percentage
            confidence = float(np.max(prediction[0])) * 100

            # Get label
            labelMap = {0: "Stand", 1: "Sit", 2: "Walk", 3: "Fall"}
            label = labelMap.get(labelIdx, "Unknown")

            # Store the prediction for next time
            HARPredictorGMM._last_prediction = (
                label, confidence, prediction[0].tolist())

            # Emit prediction result
            print(
                f"[GMM Result] {label} with {confidence:.2f}% confidence")
            self.predictionResult.emit(
                label, confidence, prediction[0].tolist())

        except Exception as e:
            print(f"Error during GMM prediction: {e}")
            self.predictionResult.emit("Error", 0.0, [0.0, 0.0, 0.0, 0.0])

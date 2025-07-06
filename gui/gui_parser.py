# ----- Imports -------------------------------------------------------

# Standard Imports
import struct
import serial
import time
import math
import datetime
import os

from PyQt5.QtCore import QThread, pyqtSignal
from collections import deque, OrderedDict
import pandas as pd
import numpy as np
# from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture

# Tensorflow
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

# Local Imports
from parseFrame import *

# other import
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


class uartParser():
    def __init__(self, preprocess=None, type='SDK Out of Box Demo'):
        # Set this option to 1 to save UART output from the radar device
        # self.saveBinary = 0
        self.preprocess = preprocess if preprocess else Preprocess()
        self.predictor = self.preprocess.predictor
        self.replay = 0
        # self.binData = bytearray(0)
        self.uartCounter = 0
        self.framesPerFile = 100
        self.first_file = True
        self.filepath = datetime.datetime.now().strftime("%m_%d_%Y_%H_%M_%S")

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

    # This function is always called - first read the UART, then call a function to parse the specific demo output
    # This will return 1 frame of data. This must be called for each frame of data that is expected. It will return a dict containing all output info
    # Point Cloud and Target structure are liable to change based on the lab. Output is always cartesian.
    # DoubleCOMPort means this function refers to the xWRx843 family of devices.
    def readAndParseUartDoubleCOMPort(self):

        self.fail = 0
        if (self.replay):
            return self.replayHist()

        # Find magic word, and therefore the start of the frame
        index = 0
        try:
            magicByte = self.dataCom.read(1)
        except Exception as e:
            print(f"Serial error: {e}")

        frameData = bytearray(b'')
        while (1):
            # If the device doesnt transmit any data, the COMPort read function will eventually timeout
            # Which means magicByte will hold no data, and the call to magicByte[0] will produce an error
            # This check ensures we can give a meaningful error
            if (len(magicByte) < 1):
                print("ERROR: No data detected on COM Port, read timed out")
                print(
                    "\tBe sure that the device is in the proper mode, and that the cfg you are sending is valid")
                magicByte = self.dataCom.read(1)

            # Found matching byte
            elif (magicByte[0] == UART_MAGIC_WORD[index]):
                index += 1
                frameData.append(magicByte[0])
                if (index == 8):  # Found the full magic word
                    break
                magicByte = self.dataCom.read(1)

            else:
                # When you fail, you need to compare your byte against that byte (ie the 4th) AS WELL AS compare it to the first byte of sequence
                # Therefore, we should only read a new byte if we are sure the current byte does not match the 1st byte of the magic word sequence
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
        # This ensures that we only read the part of the frame in that we are lacking
        frameLength -= 16

        # Read in rest of the frame
        frameData += bytearray(self.dataCom.read(frameLength))

        # frameData now contains an entire frame, send it to parser
        if (self.parserType == "DoubleCOMPort"):
            outputDict = parseStandardFrame(frameData)

            # Ngeprint Point Cloud
            points = outputDict.get('pointCloud', [])
            # print("=== Point Cloud Data ===")
            # if (isinstance(points, np.ndarray) and points.size > 0) or (isinstance(points, list) and len(points) > 0):
            #     for point in points:
            #         print(point)
            # else:
            #     print("No point cloud data found in this frame.")

            self.preprocess.printPointCloud(points)

        else:
            print('FAILURE: Bad parserType')

        return outputDict

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

    # send single command to device over UART Com.

    def sendLine(self, line):
        self.cliCom.write(line.encode())
        ack = self.cliCom.readline()
        print(ack)
        ack = self.cliCom.readline()
        print(ack)


def getBit(byte, bitNum):
    mask = 1 << bitNum
    if (byte & mask):
        return 1
    else:
        return 0


class Preprocess():
    def __init__(self, predictor=None):
        super().__init__()
        self.predictor = predictor if predictor else Predictor()
        self.pointCloudBuffer = []
        self.uniqueTimestamps = set()
        self._stride_counter = 0
        self.stride_value = 30
        self.maxTimestamps = 30
        self.collected_data = None
        self._start_time = None

        # New attributes for sliding window
        # Keep only last 30 timestamps
        self.timestampBuffer = deque(maxlen=self.maxTimestamps)
        self.timestampDataMap = OrderedDict()  # Map timestamp to its data points
        self.windowGroups = []  # Store all generated windows

        # Connect signals from predictor thread
        self.predictor.processing_finished.connect(self.on_processing_finished)
        self.predictor.processing_error.connect(self.on_processing_error)
        self.predictor.processing_started.connect(self.on_processing_started)

    def printPointCloud(self, points):
        timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S.%f')
        outputWithTimestamp = []
        numFrame = 1

        if isinstance(points, np.ndarray) and points.size > 0:
            for point in points:
                row = [timestamp, numFrame] + \
                    [round(float(x), 4) for x in point[:5]]
                outputWithTimestamp.append(row)
                numFrame += 1
        # else:
        #     print(f"{timestamp} Tidak tidak memiliki point cloud yang valid.")

        if outputWithTimestamp:
            self.addToSlidingWindow(outputWithTimestamp)
            # self.start_time = datetime.datetime.now()
        return outputWithTimestamp

    def addToSlidingWindow(self, dataFormatted):
        if not dataFormatted:
            return

        currentTimestamp = dataFormatted[0][0]

        # Skip jika timestamp sudah pernah masuk
        if currentTimestamp in self.timestampDataMap:
            return

        # Tambahkan timestamp baru ke buffer dan mapping
        self.timestampBuffer.append(currentTimestamp)
        self.timestampDataMap[currentTimestamp] = dataFormatted

        # Buang timestamp lama jika sudah melebihi kapasitas
        if len(self.timestampDataMap) > self.maxTimestamps:
            oldestTimestamp = next(iter(self.timestampDataMap))
            del self.timestampDataMap[oldestTimestamp]

        # Tambahkan hitungan stride
        self._stride_counter += 1

        # Cek apakah sudah cukup timestamp
        if len(self.timestampBuffer) >= self.maxTimestamps:
            if self._stride_counter >= self.stride_value:
                self.createSlidingWindow()
                self._stride_counter = 0  # reset counter setelah membuat window

    def createSlidingWindow(self):
        try:
            # Get current window (latest 30 timestamps)
            currentWindow = list(self.timestampBuffer)
            windowData = []

            # Collect all data points for this window
            for timestamp in currentWindow:
                if timestamp in self.timestampDataMap:
                    windowData.extend(self.timestampDataMap[timestamp])

            if windowData:
                # Convert to numpy array
                windowArray = np.array(windowData)

                # Store this window
                windowInfo = {
                    'timestamps': currentWindow.copy(),
                    'data': windowArray,
                    'window_id': len(self.windowGroups) + 1,
                    'first_timestamp': currentWindow[0],
                    'last_timestamp': currentWindow[-1]
                }

                self.windowGroups.append(windowInfo)

                # print(f"Window timestamps: {currentWindow[0]} to {currentWindow[-1]}")

                # Update collected_data to latest window
                self.collected_data = windowArray

                window_start_time = datetime.datetime.now()
                if not self.predictor.isRunning():
                    self.predictor.setData(windowArray, window_start_time)
                    self.predictor.start()
                else:
                    self.predictor.queueData(windowArray, window_start_time)

                return True

        except Exception as e:
            print(f"Error creating sliding window: {str(e)}")
            return False

    # Signal handlers for thread communication
    def on_processing_started(self):
        print("Processing started in background thread...")

    def on_processing_finished(self, processed_data):
        print("Processing completed in background thread\n")
        if processed_data is not None:
            print(f"Processed data shape: {processed_data.shape}")
            # Update collected_data with processed result if needed
            # self.collected_data = processed_data

    def on_processing_error(self, error_message):
        print(f"Processing error in background thread: {error_message}")

    def getLatestWindow(self):
        return self.collected_data

    def getAllWindows(self):
        return self.windowGroups

    def getWindow(self, window_id):
        if 0 < window_id <= len(self.windowGroups):
            return self.windowGroups[window_id - 1]
        return None

    def getCurrentBufferStatus(self):
        return {
            'currentTimestamps': len(self.timestampBuffer),
            'totalDataPoints': sum(len(data) for data in self.timestampDataMap.values()),
            'targetTimestamps': self.maxTimestamps,
            'windowsGenerated': len(self.windowGroups),
            'isReady': len(self.timestampBuffer) >= self.maxTimestamps,
            'completionPercentage': (len(self.timestampBuffer) / self.maxTimestamps) * 100,
            'thread_running': self.predictor.isRunning()
        }

    def resetBuffer(self):
        # Stop thread if running
        if self.predictor.isRunning():
            self.predictor.stop()
            self.predictor.wait()  # Wait for thread to finish

        self.pointCloudBuffer = []
        self.uniqueTimestamps = set()
        self.timestampBuffer.clear()
        self.timestampDataMap.clear()
        self.windowGroups = []
        self.collected_data = None
        print("All buffers reset. Ready for new data collection.")

    # Legacy methods for backward compatibility
    def addToBuffer(self, dataFormatted):
        self.addToSlidingWindow(dataFormatted)

    def convertBufferToNumpyArray(self):
        return len(self.windowGroups) > 0

    def getCollectedData(self):
        return self.getLatestWindow()


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


# class TNet(layers.Layer):
#     def __init__(self, k, reg_factor=0.001, **kwargs):  # Tambahkan **kwargs untuk menangani argumen tambahan
#         super(TNet, self).__init__(**kwargs)  # Pastikan super() menangani kwargs
#         self.k = k
#         self.reg_factor = reg_factor
#         self.conv1 = layers.Conv1D(64, 1, activation='relu')
#         self.conv2 = layers.Conv1D(128, 1, activation='relu')
#         self.conv3 = layers.Conv1D(1024, 1, activation='relu')
#         self.fc1 = layers.Dense(512, activation='relu')
#         self.fc2 = layers.Dense(256, activation='relu')
#         self.fc3 = layers.Dense(k * k, activation=None, kernel_initializer='zeros')
#         self.batch_norm1 = layers.BatchNormalization()
#         self.batch_norm2 = layers.BatchNormalization()
#         self.batch_norm3 = layers.BatchNormalization()
#         self.batch_norm4 = layers.BatchNormalization()
#         self.batch_norm5 = layers.BatchNormalization()


#     def build(self, input_shape):
#         self.built = True

#     def call(self, inputs):
#         x = self.conv1(inputs)
#         x = self.batch_norm1(x)
#         x = self.conv2(x)
#         x = self.batch_norm2(x)
#         x = self.conv3(x)
#         x = self.batch_norm3(x)
#         x = layers.GlobalMaxPooling1D()(x)
#         x = self.fc1(x)
#         x = self.batch_norm4(x)
#         x = self.fc2(x)
#         x = self.batch_norm5(x)
#         x = self.fc3(x)

#         x = tf.reshape(x, (-1, self.k, self.k))
#         identity = tf.eye(self.k, batch_shape=[tf.shape(inputs)[0]])
#         x = x + identity

#         x_transpose = tf.transpose(x, perm=[0, 2, 1])
#         product = tf.matmul(x, x_transpose, transpose_b=True)
#         orth_loss = self.reg_factor * tf.reduce_mean(tf.square(product - identity))
#         self.add_loss(orth_loss)

#         transformed = tf.matmul(inputs, x)
#         return transformed


get_custom_objects().update({"TNet": TNet})


class Predictor(QThread):
    # Define signals for thread communication
    processing_started = pyqtSignal()
    processing_finished = pyqtSignal(object)  # Pass processed data
    processing_error = pyqtSignal(str)  # Pass error message
    predictionResult = pyqtSignal(str, float, list)

    _loadedModel = None
    _lastPrediction = None

    def __init__(self, modelPath=None, type=None):
        super().__init__()
        self.windowArray = None
        # Thread control
        self._stop_flag = False
        self._data_queue = deque()
        self.type = type

        # Classification
        self.modelPath = modelPath or os.path.abspath(os.path.join(
            os.path.dirname(__file__), "..", "models", "Bootstrap-Balanced2.h5"))
        # self.modelPath = modelPath or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "Bootstrap-Balanced6.h5"))

        # Load model at initialization to avoid delay on first prediction
        if Predictor._loadedModel is None:
            try:
                Predictor._loadedModel = load_model(self.modelPath, custom_objects={
                                                    "TNet": TNet}, compile=False)
                print(f"Model berhasil di Load (awal)\n")
            except Exception as e:
                print(f"Error loading PointNet model: {e}\n")

    def setData(self, windowArray, start_time=None):
        self.windowArray = windowArray.copy() if windowArray is not None else None
        self._start_time = start_time or datetime.datetime.now()

    def queueData(self, windowArray, start_time=None):
        if windowArray is not None:
            self._data_queue.append(
                (windowArray.copy(), start_time or datetime.datetime.now()))

    def stop(self):
        self._stop_flag = True

    def run(self):
        try:
            self._stop_flag = False

            while not self._stop_flag:
                # Process current data
                if self.windowArray is not None:
                    self.processing_started.emit()
                    result = self.checkBufferSize(
                        self.windowArray, self._start_time)
                    self.processing_finished.emit(result)
                    self.windowArray = None

                # Process queued data
                if self._data_queue and not self._stop_flag:
                    next_data, next_start_time = self._data_queue.popleft()
                    if next_data is not None:
                        self.processing_started.emit()
                        self._start_time = next_start_time  # ini yang penting!
                        result = self.checkBufferSize(next_data)
                        self.processing_finished.emit(result)

                else:
                    # No more data to process, exit thread
                    break

        except Exception as e:
            error_msg = f"Thread execution error: {str(e)}"
            print(error_msg)
            self.processing_error.emit(error_msg)

    def checkBufferSize(self, windowArray, start_time=None):

        if self._stop_flag:
            return None

        self.pointCloudArray = windowArray.copy()
        self._start_time = start_time or datetime.datetime.now()

        # Konversi seluruh array menjadi float64 terlebih dahulu
        try:
            # Method 1: Konversi langsung seluruh array
            self.pointCloudArray = self.pointCloudArray.astype(np.float64)

        except ValueError as e:
            print(f"Direct conversion failed: {e}")

            # Method 2: Konversi kolom per kolom jika direct conversion gagal
            try:
                converted_array = np.zeros(
                    (self.pointCloudArray.shape[0], self.pointCloudArray.shape[1]), dtype=np.float64)

                for i in range(self.pointCloudArray.shape[1]):
                    if self._stop_flag:
                        return None
                    # Konversi setiap kolom ke float64
                    converted_array[:, i] = pd.to_numeric(
                        self.pointCloudArray[:, i], errors='coerce').astype(np.float64)

                self.pointCloudArray = converted_array
                print(f"Column-wise conversion successful")
                print(f"Array dtype: {self.pointCloudArray.dtype}")

            except Exception as e2:
                print(f"Column-wise conversion also failed: {e2}")
                return None

        # Validasi bahwa semua data sudah numeric
        if not np.issubdtype(self.pointCloudArray.dtype, np.number):
            print(
                f"Warning: Array is still not numeric. Current dtype: {self.pointCloudArray.dtype}")
            return None

        if self._stop_flag:
            return None

        self.filteredData = self.pointCloudArray
        return self.bootstrap(self.pointCloudArray)

    def bootstrap(self, filteredData, targetCount=150, nComponents=5, minPartial=8):
        try:
            if self._stop_flag or filteredData is None or len(filteredData) == 0:
                return None

            print(f"Input to Bootstrap: {filteredData.shape}")

            # Kolom 0: timestamp, Kolom 1: numFrame, Kolom 2-4: x,y,z, Kolom 5-6: doppler,snr
            if filteredData.shape[1] < 7:
                print(
                    f"Error: Insufficient columns for bootstrap. Expected 7, got {filteredData.shape[1]}")
                return filteredData

            # Ekstrak timestamp unik dan hitung ukuran grup
            timestamps = filteredData[:, 0]  # Kolom 0 adalah timestamp
            uniqueTimestamps, counts = np.unique(
                timestamps, return_counts=True)

            # Cari timestamp dengan jumlah point terbanyak sebagai reference
            referenceTimestamp = uniqueTimestamps[np.argmax(counts)]
            referenceIndices = timestamps == referenceTimestamp
            # x, y, z, doppler, snr (kolom 2-6)
            referenceData = filteredData[referenceIndices, 2:7]

            processedList = []
            totalProcessed = 0

            for timestamp in uniqueTimestamps:
                if self._stop_flag:
                    break

                # Ambil data untuk timestamp ini
                timestampIndices = timestamps == timestamp
                timestampData = filteredData[timestampIndices]
                groupSize = len(timestampData)
                # print(groupSize)
                # Skip jika sudah cukup atau terlalu sedikit
                if groupSize >= targetCount:
                    processedList.append(timestampData[:targetCount])
                    continue

                if groupSize < max(minPartial, nComponents):
                    # x, y, z, doppler, snr
                    partialPoints = timestampData[:, 2:7]
                    completedPoints = self.duplicatePointCloud(
                        partialPoints, targetCount)
                    if completedPoints is not None and not self._stop_flag:
                        completedArray = np.zeros(
                            (len(completedPoints), filteredData.shape[1]))
                        completedArray[:, 0] = timestamp
                        completedArray[:, 1] = np.arange(
                            1, len(completedPoints) + 1)
                        completedArray[:, 2:7] = completedPoints
                        processedList.append(completedArray)
                        totalProcessed += 1
                    continue

                partialPoints = timestampData[:, 2:7]
                completedPoints = self.performGMM(
                    referenceData, partialPoints, targetCount, nComponents)
                if completedPoints is not None and not self._stop_flag:
                    completedArray = np.zeros(
                        (len(completedPoints), filteredData.shape[1]))
                    completedArray[:, 0] = timestamp
                    completedArray[:, 1] = np.arange(
                        1, len(completedPoints) + 1)
                    completedArray[:, 2:7] = completedPoints
                    processedList.append(completedArray)
                    # print(completedArray.shape)
                    totalProcessed += 1

            if processedList and not self._stop_flag:
                result = np.vstack(processedList)
                # print(f"Bootstrap completed: {totalProcessed} timestamps processed, total {len(result)} points")
                print(f"Shape after Bootstrap: {result.shape}")
                arrayData = result
                self.performModel(arrayData)
                return result
            else:
                return filteredData if not self._stop_flag else None

        except Exception as e:
            if not self._stop_flag:
                print(f"Error in bootstrap: {str(e)}")
                import traceback
                traceback.print_exc()
            return filteredData

    def performGMM(self, referenceData, partialPoints, targetCount=150, nComponents=5):
        try:
            if self._stop_flag:
                return None

            if len(referenceData) < nComponents:
                nComponents = max(1, len(referenceData))

            gmm = GaussianMixture(
                n_components=nComponents,
                covariance_type='full',
                random_state=42,
                max_iter=100
            )

            gmm.fit(referenceData)

            if self._stop_flag:
                return None

            toSample = targetCount - len(partialPoints)

            if toSample <= 0:
                return partialPoints[:targetCount]

            sampledPoints = gmm.sample(n_samples=toSample)[0]
            completed = np.vstack([partialPoints, sampledPoints])

            return completed if not self._stop_flag else None

        except Exception as e:
            if not self._stop_flag:
                print(f"Error in performGMM: {str(e)}")
            return partialPoints

    def duplicatePointCloud(self, pointCloud, targetPoints):
        numOriginalPoints = pointCloud.shape[0]
        if numOriginalPoints >= targetPoints:
            return pointCloud[:targetPoints]
        indices = np.random.choice(
            numOriginalPoints, size=targetPoints, replace=True)
        bootstrapped_cloud = pointCloud[indices]
        return bootstrapped_cloud

    def performModel(self, arrayData):
        # print(f"Input shape before drop column: {arrayData.shape}")
        if Predictor._loadedModel is None:
            print("Model belum berhasil di-load!")
            return

        self.model = Predictor._loadedModel

        # Drop kolom index 0, dan 1 yang tidak dipakai
        processedData = np.delete(
            arrayData, [0, 1, 6], axis=1)  # SNR lagi mati
        print(f"Model input Shape: {processedData.shape}")

        expectedPoint = 4500
        currentPoint = processedData.shape[0]
        SNR = processedData[:, 4].min()

        if currentPoint == expectedPoint:
            try:
                if len(processedData.shape) == 2:
                    processedData = np.expand_dims(processedData, axis=0)

                prediction = self.model.predict(processedData, verbose=0)

                labelIdx = np.argmax(prediction)
                confidence = float(np.max(prediction[0])) * 100

                labelMap = {0: "Stand", 1: "Sit", 2: "Walk", 3: "Fall"}
                label = labelMap.get(labelIdx, "Unknown")

                # print(dopplerAverage)

                # if self._start_time:
                #     latency = (datetime.datetime.now() - self._start_time).total_seconds()
                #     print(f"Latency dari windowing ke : {latency:.4f} detik")
                #     self._start_time = None

                # print("Probabilitas tiap kelas:")
                # for idx, prob in enumerate(prediction[0]):
                #     kelas = labelMap.get(idx, f"Kelas-{idx}")
                #     print(f"  {kelas}: {prob * 100:.2f}%")

                # Predictor._lastPrediction = (label, confidence, prediction[0].tolist())

                # print(f"Prediction {label} with {confidence:.2f}% confidence\n")
                # self.predictionResult.emit(label, confidence, prediction[0].tolist())

                finalLabel = label  # Siapkan label akhir
                SNR = abs(SNR)
                print(SNR)

                if label == "Fall":
                    if 10 < SNR <= 50:
                        finalLabel = "Walk"
                    elif SNR <= 10:
                        finalLabel = "Stand"

                if self._start_time:
                    latency = (datetime.datetime.now() -
                               self._start_time).total_seconds()
                    print(
                        f"Latency dari windowing ke prediksi: {latency:.4f} detik")
                    self._start_time = None

                print("Probabilitas tiap kelas:")
                for idx, prob in enumerate(prediction[0]):
                    kelas = labelMap.get(idx, f"Kelas-{idx}")
                    print(f"  {kelas}: {prob * 100:.2f}%")

                # Simpan dan kirim hasil prediksi AKHIR
                Predictor._lastPrediction = (
                    finalLabel, confidence, prediction[0].tolist())

                print(
                    f"Prediction: {finalLabel} with {confidence:.2f}% confidence\n")
                self.predictionResult.emit(
                    finalLabel, confidence, prediction[0].tolist())
            except Exception as e:
                print(f"Error during prediction: {e}")

        else:
            print(f"Jumlah point tidak cukup\n")

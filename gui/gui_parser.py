# ----- Imports -------------------------------------------------------

# Standard Imports
import struct
import serial
import time
import numpy as np
import math
import datetime
import threading
import os

# Local Imports
from parseFrame import *
from PyQt5.QtCore import QThread, pyqtSignal

#Initialize this Class to create a UART Parser. Initialization takes one argument:
# 1: String Lab_Type - These can be:
#   a. 3D People Tracking
#   b. SDK Out of Box Demo
#   c. Long Range People Detection
#   d. Indoor False Detection Mitigation
#   e. (Legacy): Overhead People Counting
#   f. (Legacy) 2D People Counting
# Default is (f). Once initialize, call connectComPorts(self, cliComPort, DataComPort) to connect to device com ports.
# Then call readAndParseUart() to read one frame of data from the device. The gui this is packaged with calls this every frame period.
# readAndParseUart() will return all radar detection and tracking information.
class uartParser():
    def __init__(self,type='SDK Out of Box Demo'):
        # Set this option to 1 to save UART output from the radar device
        self.currentThread = None
        self.saveBinary = 0
        self.replay = 0
        self.binData = bytearray(0)
        self.uartCounter = 0
        self.framesPerFile = 100
        self.first_file = True
        self.filepath = datetime.datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
        self.batches = []
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
            print ("ERROR, unsupported demo type selected!")
        
        # Data storage
        self.now_time = datetime.datetime.now().strftime('%Y%m%d-%H%M')
    
    def disconnectComPort(self):
        try:
            if self.currentThread is not None:
                if self.currentThread.isRunning():
                    self.currentThread.quit()
                    self.currentThread.wait()
            if hasattr(self, 'cliPort') and self.cliPort is not None:
                self.cliPort.close()
                self.cliPort = None
            if hasattr(self, 'dataPort') and self.dataPort is not None:
                self.dataPort.close()
                self.dataPort = None
            print("Serial ports disconnected.")
        except Exception as e:
            print("Error disconnecting COM ports:", e)


    def WriteFile(self, data):
        filepath=self.now_time + '.bin'
        objStruct = '6144B'
        objSize = struct.calcsize(objStruct)
        binfile = open(filepath, 'ab+') #open binary file for append
        binfile.write(bytes(data))
        binfile.close()

    def setSaveBinary(self, saveBinary):
        self.saveBinary = saveBinary
        print(self.saveBinary)

    # This function is always called - first read the UART, then call a function to parse the specific demo output
    # This will return 1 frame of data. This must be called for each frame of data that is expected. It will return a dict containing all output info
    # Point Cloud and Target structure are liable to change based on the lab. Output is always cartesian.
    # DoubleCOMPort means this function refers to the xWRx843 family of devices.

    
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

        # If save binary is enabled
        if (self.saveBinary == 1):
            self.binData += frameData
            self.uartCounter += 1
            if (self.uartCounter % self.framesPerFile == 0):
                if (self.first_file is True):
                    if (os.path.exists('binData/') == False):
                        os.mkdir('binData/')
                    os.mkdir('binData/' + self.filepath)
                    self.first_file = False
                toSave = bytes(self.binData)
                fileName = 'binData/' + self.filepath + '/pHistBytes_' + str(math.floor(self.uartCounter/self.framesPerFile)) + '.bin'
                bfile = open(fileName, 'wb')
                bfile.write(toSave)
                bfile.close()
                self.binData = []

        # Make sure outputDict is assigned properly.
        if (self.parserType == "DoubleCOMPort"):
            outputDict = parseStandardFrame(frameData)
            
            # Debugging
            if 'pointCloud' in outputDict:
                print(f"PointCloud ditemukan, jumlah point: {len(outputDict['pointCloud'])}")
            else:
                print("❌ PointCloud TIDAK ditemukan dalam outputDict.")
        else:
            print('FAILURE: Bad parserType')

        # Ensure point cloud data exists before trying to access it
        pointCloudData = outputDict.get('pointCloud', [])
        
        if pointCloudData is not None and len(pointCloudData) > 0:
            # thread = RecordPointCloudThread(pointCloudData)
            # thread.start()
            if self.currentThread is not None and self.currentThread.isRunning():
                print("🛑 Thread lama masih jalan, stop dulu.")
                self.currentThread.quit()
                self.currentThread.wait()
                # Buat thread baru
                self.currentThread = RecordPointCloudThread(pointCloudData)
                self.currentThread.start()
        return outputDict

    # This function is identical to the readAndParseUartDoubleCOMPort function, but it's modified to work for SingleCOMPort devices in the xWRLx432 family
    def readAndParseUartSingleCOMPort(self):
        # Reopen CLI port
        if(self.cliCom.isOpen() == False):
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
                print ("ERROR: No data detected on COM Port, read timed out")
                print("\tBe sure that the device is in the proper mode, and that the cfg you are sending is valid")
                magicByte = self.cliCom.read(1)

            # Found matching byte
            elif (magicByte[0] == UART_MAGIC_WORD[index]):
                index += 1
                frameData.append(magicByte[0])
                if (index == 8): # Found the full magic word
                    break
                magicByte = self.cliCom.read(1)
                
            else:
                # When you fail, you need to compare your byte against that byte (ie the 4th) AS WELL AS compare it to the first byte of sequence
                # Therefore, we should only read a new byte if we are sure the current byte does not match the 1st byte of the magic word sequence
                if (index == 0):
                    magicByte = self.cliCom.read(1)
                index = 0 # Reset index
                frameData = bytearray(b'') # Reset current frame data
        
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

        # If save binary is enabled
        if(self.saveBinary == 1):
            self.binData += frameData
            # Save data every framesPerFile frames
            self.uartCounter += 1
            if (self.uartCounter % self.framesPerFile == 0):
                # First file requires the path to be set up
                if(self.first_file is True): 
                    if(os.path.exists('binData/') == False):
                        # Note that this will create the folder in the caller's path, not necessarily in the Industrial Viz Folder                        
                        os.mkdir('binData/')
                    os.mkdir('binData/'+self.filepath)
                    self.first_file = False
                toSave = bytes(self.binData)
                fileName = 'binData/'+self.filepath+'/pHistBytes_'+str(math.floor(self.uartCounter/self.framesPerFile))+'.bin'
                bfile = open(fileName, 'wb')
                bfile.write(toSave)
                bfile.close()
                # Reset binData and missed frames
                self.binData = []
                
        # frameData now contains an entire frame, send it to parser
        if (self.parserType == "SingleCOMPort"):
            outputDict = parseStandardFrame(frameData)
        else:
            print ('FAILURE: Bad parserType')
        
        return outputDict


    # Find various utility functions here for connecting to COM Ports, send data, etc...
    # Connect to com ports
    # Call this function to connect to the comport. This takes arguments self (intrinsic), cliCom, and dataCom. No return, but sets internal variables in the parser object.
    def connectComPorts(self, cliCom, dataCom):
        self.cliCom = serial.Serial(cliCom, 115200, parity=serial.PARITY_NONE, stopbits=serial.STOPBITS_ONE, timeout=0.6)
        self.dataCom = serial.Serial(dataCom, 921600, parity=serial.PARITY_NONE, stopbits=serial.STOPBITS_ONE, timeout=0.6)
        self.dataCom.reset_output_buffer()
        print('Connected')
    
    # Separate connectComPort (not PortS) for IWRL6432 because it only uses one port
    def connectComPort(self, cliCom, cliBaud=115200):
        # Longer timeout time for IWRL6432 to support applications with low power / low update rate
        self.cliCom = serial.Serial(cliCom, cliBaud, parity=serial.PARITY_NONE, stopbits=serial.STOPBITS_ONE, timeout=4)
        self.cliCom.reset_output_buffer()
        print('Connected (one port)')

    #send cfg over uart
    def sendCfg(self, cfg):
        # Ensure each line ends in \n for proper parsing
        for i, line in enumerate(cfg):
            # Remove empty lines from cfg
            if(line == '\n'):
                cfg.remove(line)
            # add a newline to the end of every line (protects against last line not having a newline at the end of it)
            elif(line[-1] != '\n'):
                cfg[i] = cfg[i] + '\n'

        for line in cfg:
            time.sleep(.03) # Line delay

            if(self.cliCom.baudrate == 1250000):
                for char in [*line]:
                    time.sleep(.001) # Character delay. Required for demos which are 1250000 baud by default else characters are skipped
                    self.cliCom.write(char.encode())
            else:
                self.cliCom.write(line.encode())
                
            ack = self.cliCom.readline()
            print(ack)
            ack = self.cliCom.readline()
            print(ack)
            splitLine = line.split()
            if(splitLine[0] == "baudRate"): # The baudrate CLI line changes the CLI baud rate on the next cfg line to enable greater data streaming off the IWRL device.
                try:
                    self.cliCom.baudrate = int(splitLine[1])
                except:
                    print("Error - Invalid baud rate")
                    sys.exit(1)
        # Give a short amount of time for the buffer to clear
        time.sleep(0.03)
        self.cliCom.reset_input_buffer()
        # NOTE - Do NOT close the CLI port because 6432 will use it after configuration


    #send single command to device over UART Com.
    def sendLine(self, line):
        self.cliCom.write(line.encode())
        ack = self.cliCom.readline()
        print(ack)
        ack = self.cliCom.readline()
        print(ack)

    
class RecordPointCloudThread(QThread):
    def __init__(self, pointCloudData):
        QThread.__init__(self)
        self.pointCloudData = pointCloudData
        self._running = True
        print("[Thread] Initialized")

    def run(self):
        # Logika perekaman 5 detik ada di sini
        start_time = time.perf_counter()
        collected_data = []

        while time.perf_counter() - start_time < 5 and self._running:
            if isinstance(self.pointCloudData, np.ndarray) and self.pointCloudData.size > 0:
                self.pointCloudData = self.pointCloudData.tolist()

            if isinstance(self.pointCloudData, list) and len(self.pointCloudData) > 0:
                current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
                for point in self.pointCloudData:
                    if len(point) >= 7:
                        collected_data.append([current_time] + point[:3])

        if collected_data:
            batch_array = np.array(collected_data, dtype=object)
            print(f"✅ Batch selesai: {batch_array.shape}")
            # Disini nanti bisa dipanggil fungsi checkBatchContent atau prediksi
            self.checkBatchContent(batch_array) 
        
    
    def checkBatchContent(self, batch_array, show_rows=40):
        print("\n🧪 [checkBatchContent] Cek isi batch_array:")

        if batch_array is None:
            print("❌ batch_array is None.")
            return

        if not isinstance(batch_array, np.ndarray):
            print(f"❌ Tipe data salah: {type(batch_array)}. Seharusnya numpy.ndarray.")
            return

        if batch_array.size == 0:
            print("⚠️ batch_array kosong.")
            return

        print(f"✅ Shape: {batch_array.shape}")
        print(f"✅ Tipe per elemen: {type(batch_array[0][0])}")
        # print(f"✅ Kolom: ['Timestamp', 'x', 'y', 'z', 'Doppler', 'SNR', 'Noise', 'TrackIndex']")
        timestamps = batch_array[:, 0]
        unique_timestamps = np.unique(timestamps)
        print(f"✅ Jumlah Unique Timestamps: {len(unique_timestamps)}")
        print(f"\nContoh {min(show_rows, batch_array.shape[0])} baris pertama:")

        for i in range(min(show_rows, batch_array.shape[0])):
            print(f"{i+1}: {batch_array[i]}")

    def stop(self):
        self._running = False
        
def getBit(byte, bitNum):
    mask = 1 << bitNum
    if (byte&mask):
        return 1
    else:
        return 0
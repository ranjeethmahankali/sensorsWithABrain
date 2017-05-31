from pyfirmata import Arduino, util
import time
import random
import numpy as np

# initializing the arduino
# B = Arduino('COM3')
print("Connecting to the Arduino...")
B = Arduino('/dev/cu.usbserial-DN02PAGM')
# this is the pin used to tell the brain that I am here - the truth value
# to train against. so set this pin to high when u r here at the DMG and working
IAM_HERE_PIN = B.get_pin('d:2:i')

it = util.Iterator(B)
it.start()
# the dictionary of all the sensor instances initialized till now
SENSOR = dict()

READ_NUM = 10
READ_INTERVAL = 0.025

# returns a true if the pin num is not being used by any sensor
def pinAvailable(num):
    available = True
    for name in SENSOR:
        if SENSOR[name].pin == num:
            available = False
            break
    
    return available

# this is a class for sensor
class sensor:
    def __init__(self, sensorName, pin_number):
        # if the name or the pin_number are already used then aborting
        if (sensorName in SENSOR) or (not pinAvailable(pin_number)):
            raise ValueError("sensor name or pin number is repeated !")
        
        # creating a sensor instance
        self.name = sensorName
        self.pin = pin_number
        # enabling reporting for this sensor
        B.analog[self.pin].enable_reporting()
        # adding the sensor to the dictionary of sensors
        SENSOR[self.name] = self
    
    # returns the length of data in a single read for this sensor
    def read_length(self):
        interval = 1/self.frequency
        count = int(self.duration//interval)+1 # using integer division for rounding off
        return count

    # returns a single reading from sensor
    def reading(self):
        read = B.analog[self.pin].read()
        if not read is None:
            return read
        else:
            time.sleep(0.001)
            return self.reading()
    # reads the data from the sensor and returns it as an array
    # readings are taken for appropriate duration at appropriate frequency

# this is a class for a motor - a motor nerve control for the NN
class motor:
    def __init__(self, motorName, pin_number):
        self.pin_num  = pin_number
        self.pin = B.get_pin(self.pin_num)
    
    def write(self, value):
        self.pin.write(value)
# returns all the sensed data
def allSensors_read():
    data = []
    for i in range(READ_NUM):
        for name in SENSOR:
            data.append(SENSOR[name].reading())
        time.sleep(READ_INTERVAL)
    
    return data

# returns the total length of the data from all sensors
def data_length_all():
     return len(SENSOR)*READ_NUM

# this is where the main code begins - for testing
# temp = sensor("temp", 0)
# light = sensor("light",1)
# dist = sensor("distance", 2)
# head = motor("head", 9)

# while True:
#     data = allSensors_read()
#     print(data)
#     rand = random.random()
#     dataArr = np.array(data)
    
#     head.write(180*rand*dataArr.mean())
#     time.sleep(0.01)
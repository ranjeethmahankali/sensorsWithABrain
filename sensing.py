from pyfirmata import Arduino, util
import time
import random
import numpy as np

# initializing the arduino
B = Arduino('/dev/cu.usbserial-DN02PAGM')
# B = Arduino('COM3')
it = util.Iterator(B)
it.start()
# the dictionary of all the sensor instances initialized till now
SENSOR = dict()

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
    def __init__(self, sensorName, pin_number, read_frequency = 5, read_duration = 1):
        # if the name or the pin_number are already used then aborting
        if (sensorName in SENSOR) or (not pinAvailable(pin_number)):
            raise ValueError("sensor name or pin number is repeated !")
        
        # creating a sensor instance
        self.name = sensorName
        self.pin = pin_number
        # this is the read frequency of the sensor
        self.frequency = read_frequency
        # this is the duration for which the sensor will collect data in a single read
        self.duration = read_duration
        # enabling reporting for this sensor
        B.analog[self.pin].enable_reporting()
        # adding the sensor to the dictionary of sensors
        SENSOR[self.name] = self

    # returns a single reading from sensor
    def single_reading(self):
        read = B.analog[self.pin].read()
        if not read is None:
            return read
        else:
            time.sleep(0.001)
            return self.single_reading()
    # reads the data from the sensor and returns it as an array
    # readings are taken for appropriate duration at appropriate frequency
    def read(self):
        interval = 1/self.frequency
        count = int(self.duration//interval)+1 # using integer division for rounding off
        data = []
        for _ in range(count):
            data.append(self.single_reading())
            time.sleep(interval)
        
        return data

# this is a class for a motor - a motor nerve control for the NN
class motor:
    def __init__(self, motorName, pin_number):
        self.pin_num  = pin_number
        self.pin = B.get_pin('d:%s:s'%self.pin_num)
    
    def write(self, value):
        self.pin.write(value)
# returns all the sensed data
def allSensors_read():
    data = []
    for name in SENSOR:
        data += SENSOR[name].read()
    
    return data

# this is where the main code begins - for testing
temp = sensor("temp", 0)
light = sensor("light",1)
dist = sensor("distance", 2)
head = motor("head", 9)

while True:
    data = allSensors_read()
    print(data)
    rand = random.random()
    dataArr = np.array(data)
    
    head.write(180*rand*dataArr.mean())
    time.sleep(0.01)
from pyfirmata import Arduino, util

# initializing the arduino
# B = Arduino('/dev/cu.usbserial-DN02PAGM')
# B = Arduino('COM3')
# the dictionary of all the sensor instances initialized till now
SENSOR = dict()

# returns a true if the pin num is not being used by any sensor
def pinAvailable(num):
    available = True
    for name in SENSOR:
        if SENSOR[name] == num:
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
        # B.analog[self.pin].enable_reporting()
        # adding the sensor to the dictionary of sensors
        SENSOR[self.name] = self.pin

    # returns a single reading from sensor
    def single_reading(self):
        read = B.analog[self.pin].read()
        if not read is None:
            return read
        else:
            time.sleep(0.001)
            return getReading(num)
    # reads the data from the sensor and returns it as an array
    # readings are taken for appropriate duration at appropriate frequency
    def read(self):
        interval = 1/self.frequency
        count = self.duration//interval # using integer division for rounding off
        data = []
        for _ in range(count):
            data.append(self.single_reading())
        
        return data

# this is where the main code begins - for testing
a = sensor("temp", 0)
b = sensor("light", 0)
from sensing import *
import sys

sound = sensor("sound", 3)
vibration = sensor("vibration", 4)
temp = sensor("temp", 0)
light = sensor("light",1)
dist = sensor("distance", 2)

maxlen = 20

while True:
    reading = sound.reading()
    num = int(reading*maxlen)
    bar = "#"*num + (" "*(maxlen-num))
    sys.stdout.write("%.2f: %s\r"%(reading, bar))
    time.sleep(0.1)

"""ddssdssdsf sdsfjkdsudhdhdds2awdqwq23ewasdasdasduuewu23ewd2311
"""
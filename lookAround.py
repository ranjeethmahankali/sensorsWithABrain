from sensing import *
import random

LARGE_RANGE = [0,180]
SMALL_RANGE = [0,30]

head = motor("motor","d:9:s")
light = sensor("light",1)

def clip(angle):
    return min(max(angle, 0), 180)

def decideAngle():
    roll = random.randint(0,4)
    if roll == 0:
        return 0
    elif roll == 4:
        angle = random.randint(*LARGE_RANGE)
    else:
        angle = random.randint(*SMALL_RANGE)
    
    # deciding the sign
    if random.random() > 0.5:
        sign = 1
    else:
        sign = -1
    
    return sign*angle

curAngle = 0
while True:
    curAngle = clip(curAngle + decideAngle())
    head.write(curAngle)
    # print(light.reading())
    time.sleep(1)
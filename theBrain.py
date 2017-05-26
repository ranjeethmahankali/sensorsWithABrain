from ops import *
import sensing as sense

# initializing all the sensors and motors
temp = sense.sensor("temp", 0)
light = sense.sensor("light",1)
dist = sense.sensor("distance", 2)
# sound
head = sense.motor("head", 9)

SENSOR_DATA_DIM = sense.data_length_all()
HIDDEN_DIM_1 = 256
HIDDEN_DIM_2 = 1024
HIDDEN_DIM_3 = 512
OUTPUT_DIM = 3
# it outputs two numbers, one is the hour of the day normalized
# the second number is the probability that I am in the room
# the third number is the motor signal to the servo - normalized angle

# Other constants from here
NUM_HOURS = 24

# these are all the variables in the model
with tf.variable_scope('vars'):
    wf1 = weightVariable([SENSOR_DATA_DIM, HIDDEN_DIM_1],'wf1')
    bf1 = biasVariable([HIDDEN_DIM_1], 'bf1')
    
    wf2 = weightVariable([HIDDEN_DIM_1, HIDDEN_DIM_2],'wf2')
    bf2 = biasVariable([HIDDEN_DIM_2], 'bf2')
    
    wf3 = weightVariable([HIDDEN_DIM_2, HIDDEN_DIM_3],'wf3')
    bf3 = biasVariable([HIDDEN_DIM_3], 'bf3')

    wf4 = weightVariable([HIDDEN_DIM_3, OUTPUT_DIM],'wf4')
    bf4 = biasVariable([OUTPUT_DIM], 'bf4')

    # these are the weights and biases for the time step
    wt = weightVariable([HIDDEN_DIM_3,HIDDEN_DIM_1], 'wt')
    bt = biasVariable([HIDDEN_DIM_1], 'bt')

# returns the placeholders for input
def get_placeholders():
    sensor_tensor = tf.placeholder(tf.float32, shape = [None, SENSOR_DATA_DIM])
    prev_inf_tensor = tf.placeholder(tf.float32, shape=[None, HIDDEN_DIM_3])
    truth_tensor = tf.placeholder(tf.float32, shape = [None, OUTPUT_DIM])

    return [sensor_tensor, prev_inf_tensor, truth_tensor]

# the main model
def run_model(sensor_data, prev_inference):
    # sensor data is the data from the sensor
    h1_val = tf.matmul(sensor_data, wf1) + bf1
    h1_val += tf.matmul(prev_inference, wt) + bt
    h1 = tf.nn.relu(h1_val)

    h2 = tf.nn.relu(tf.matmul(h1, wf2) + bf2)
    h3 = tf.nn.relu(tf.matmul(h2, wf3) + bf3)
    output = tf.nn.tanh(tf.matmul(h3, wf4) + bf4)

    norm = (output+1)/2
    return [norm,h3]

# currently implementing root mean squared error
def loss_optim(prediction, truth):
    split_axis = 1
    norm_hour, norm_detection, norm_motor = tf.split(split_axis, OUTPUT_DIM, prediction)

    epsilon = 1e-6
    hour = (norm_hour-epsilon)*NUM_HOURS
    detection = norm_detection - epsilon
    # in the truth data we ignore the angle
    true_hour, true_detection, _ = getInterpretation(truth)

    absDiff_h = tf.reduce_sum(tf.abs(hour - true_hour))
    absDiff_d = tf.reduce_sum(tf.abs(detection - true_detection))
    loss = absDiff_d + absDiff_h
    optim = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    return [loss, optim]

# returns the interpreted data from normalized output
def getInterpretation(output):
    split_axis = 1
    norm_hour, norm_detection, norm_motor = tf.split(split_axis, OUTPUT_DIM, output)

    epsilon = 0
    hour = tf.floor((norm_hour-epsilon)*NUM_HOURS)
    detection = tf.floor(2*(norm_detection - epsilon))
    angle = norm_motor*180

    return [hour, detection, angle]

def accuracy(output, truth):
    hour, detection, angle = getInterpretation(output)
    # in the truth daat we ignore the angle
    true_hour, true_detection, _ = getInterpretation(truth)

    h_correct = tf.cast(tf.equal(hour, true_hour), tf.float32)
    d_correct = tf.cast(tf.equal(detection, true_detection), tf.float32)

    h_acc = 100*tf.reduce_mean(h_correct)
    d_acc = 100*tf.reduce_mean(d_correct)

    total_acc = (h_acc + d_acc)/2

    return [h_acc, d_acc, total_acc]
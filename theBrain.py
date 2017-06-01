from ops import *
import sensing as sense

# initializing all the sensors and motors
temp = sense.sensor("temp", 0)
light = sense.sensor("light",1)
dist = sense.sensor("distance", 2)
sound = sense.sensor("sound", 3)
vibration = sense.sensor("vibration", 4)
# sound
head = sense.motor("head", "d:9:s")
detection_LED = sense.motor("detection_LED", "d:4:o")

print("%s sensors registered"%len(sense.SENSOR))

SENSOR_DATA_DIM = sense.data_length_all()
OUTPUT_DIM = 3
# it outputs two numbers, one is the hour of the day normalized
# the second number is the probability that I am in the room
# the third number is the motor signal to the servo - normalized angle

# this is the backprop limit into the past
TRUNC_BACKPROP_LENGTH = 20
# State size of the RNN
STATE_SIZE = 5
HIDDEN_DIM_STATE_1 = 10
HIDDEN_DIM_STATE_2 = 20
HIDDEN_DIM_1 = 512
HIDDEN_DIM_2 = 768
# Other constants from here
NUM_HOURS = 24

# these are all the variables in the model
with tf.variable_scope('vars'):
    # this is for the time step
    wt = weightVariable([STATE_SIZE+1, HIDDEN_DIM_STATE_1],'wt')
    bt = biasVariable([HIDDEN_DIM_STATE_1], 'bt')

    ws1 = weightVariable([HIDDEN_DIM_STATE_1, HIDDEN_DIM_STATE_2],'ws1')
    bs1 = biasVariable([HIDDEN_DIM_STATE_2], 'bs1')
    
    ws2 = weightVariable([HIDDEN_DIM_STATE_2, STATE_SIZE],'ws2')
    bs2 = biasVariable([STATE_SIZE], 'bs2')

    wf1 = weightVariable([SENSOR_DATA_DIM*STATE_SIZE, HIDDEN_DIM_1],'wf1')
    bf1 = biasVariable([HIDDEN_DIM_1], 'bf1')
    
    wf2 = weightVariable([HIDDEN_DIM_1, HIDDEN_DIM_2],'wf2')
    bf2 = biasVariable([HIDDEN_DIM_2], 'bf2')

    wf3 = weightVariable([HIDDEN_DIM_2, OUTPUT_DIM],'wf3')
    bf3 = weightVariable([OUTPUT_DIM],'bf3')

# running the model in single time step
def run_model(inp, cur_state):
    inp_reshape = tf.reshape(inp, [SENSOR_DATA_DIM, 1])
    state_concat = tf.concat(1, [inp_reshape, cur_state])
    h_state1 = tf.nn.relu(tf.matmul(state_concat, wt) + bt)
    
    h_state2 = tf.nn.relu(tf.matmul(h_state1, ws1)+bs1)
    next_state = tf.nn.relu(tf.matmul(h_state2, ws2)+bs2)

    h0 = tf.reshape(next_state, [1,SENSOR_DATA_DIM*STATE_SIZE])
    h1 = tf.nn.relu(tf.matmul(h0, wf1)+bf1)
    h2 = tf.nn.relu(tf.matmul(h1, wf2)+bf2)

    output = tf.matmul(h2, wf3)+bf3
    hour, detection, motor = tf.split(1, OUTPUT_DIM, output)
    hour_act = (tf.nn.tanh(hour)+1)/2
    detection_act = tf.nn.sigmoid(detection)
    motor_act = (1+tf.sin(motor*10))/2

    activated = tf.concat(1, [hour_act, detection_act, motor_act])
    return [activated, next_state]


# currently implementing root mean squared error
def loss(output_series, truth_series):
    split_axis = 1
    epsilon = 1e-6
    truths = tf.unpack(truth_series, axis=0)
    total_loss = 0
    motor_vals = []
    for i in range(len(output_series)):
        prediction = output_series[i]
        truth = tf.reshape(truths[i],[1,3])
        # print(truth.get_shape())

        norm_hour, norm_detection, norm_motor = tf.split(split_axis, OUTPUT_DIM, prediction)
        true_hour, true_detection, _ = tf.split(split_axis, OUTPUT_DIM, truth)
        motor_vals.append(norm_motor)

        absDiff_h = tf.reduce_sum(tf.abs(norm_hour - true_hour))
        absDiff_d = tf.reduce_sum(tf.abs(norm_detection - true_detection))
        loss = absDiff_d + absDiff_h
        total_loss += loss
    
    # rewarding the trait of looking around
    motor_history = tf.concat(0,motor_vals)
    mean, variance = tf.nn.moments(motor_history, axes=[0])
    # print(variance.get_shape())

    total_loss += curiosity*tf.abs(0.31-tf.reduce_mean(variance))

    w_vars = tf.trainable_variables()
    for v in w_vars:
        total_loss += alpha*tf.nn.l2_loss(v)

    optim = tf.train.AdamOptimizer(learning_rate).minimize(total_loss)
    return [total_loss, optim]

# returns the interpreted data from normalized output
def getInterpretation(output):
    split_axis = 1
    norm_hour, norm_detection, norm_motor = tf.split(split_axis, OUTPUT_DIM, output)

    epsilon = 0
    hour = tf.floor((norm_hour-epsilon)*NUM_HOURS)
    detection = tf.floor(2*(norm_detection - epsilon))
    angle = norm_motor*180

    return [hour, detection, angle]

def accuracy(output_series, truth_series):
    acc = 0
    truths = tf.unpack(truth_series, axis=0)
    for i in range(len(output_series)):
        ouput = output_series[i]
        truth = tf.reshape(truths[i],[1,3])
    
        hour, detection, angle = getInterpretation(output)
        # in the truth daat we ignore the angle
        true_hour, true_detection, _ = getInterpretation(truth)

        h_correct = tf.cast(tf.equal(hour, true_hour), tf.float32)
        d_correct = tf.cast(tf.equal(detection, true_detection), tf.float32)

        h_acc = 100*tf.reduce_mean(h_correct)
        d_acc = 100*tf.reduce_mean(d_correct)

        acc += ((h_acc + d_acc)/2)/len(output_series)

    return acc

# creating placeholders
input_series = tf.placeholder(tf.float32, [TRUNC_BACKPROP_LENGTH, SENSOR_DATA_DIM])
truth_series = tf.placeholder(tf.float32, [TRUNC_BACKPROP_LENGTH, OUTPUT_DIM])

init_state = tf.placeholder(tf.float32, [SENSOR_DATA_DIM, STATE_SIZE])

# creating the main model with back prop in time
inputs = tf.unpack(input_series, axis=0)
cur_state = init_state
output_series = []
for inp in inputs:
    output, cur_state = run_model(inp, cur_state)
    output_series.append(output)

loss_tensor, optim_t = loss(output_series, truth_series)
hour_t, detection_t, motor_t = getInterpretation(output)
accuracy_tensor = accuracy(output_series, truth_series)

# adding summaries
with tf.name_scope('data'):
    tf.summary.scalar('loss', loss_tensor)
    tf.summary.scalar('hour', tf.reduce_mean(hour_t))
    tf.summary.scalar('head_ang', tf.reduce_mean(motor_t))
    tf.summary.scalar('accuracy', accuracy_tensor)

merged = tf.summary.merge_all()
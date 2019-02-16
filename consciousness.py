from theBrain import *
from datetime import datetime
import shutil

SENSE_BUFFER = [np.zeros([TRUNC_BACKPROP_LENGTH, SENSOR_DATA_DIM]),
                    np.zeros([TRUNC_BACKPROP_LENGTH, OUTPUT_DIM])]

def truth_values(batch_size):
    # this is supposed to be high when I am here
    hour = datetime.now().hour/NUM_HOURS
    detection = int(sense.IAM_HERE_PIN.read())
    angle = 0

    return [[hour, detection, angle]]*batch_size

def create_buffer():
    global SENSE_BUFFER
    print("Creating sense buffer... please wait")
    for i in range(TRUNC_BACKPROP_LENGTH):
        updateBuffer()
        time.sleep(0.5)
    print("Buffer created.")

def updateBuffer():
    global SENSE_BUFFER
    truth = np.array(truth_values(1))
    sensed_data = np.array([sense.allSensors_read()])

    SENSE_BUFFER[0] = np.concatenate([SENSE_BUFFER[0][1:,:], sensed_data], axis=0)
    SENSE_BUFFER[1] = np.concatenate([SENSE_BUFFER[1][1:,:], truth], axis=0)



with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # creating new summary writers and deleting old summary logs
    print("Preparing to log and serve the training summaries to tensorboard...")
    shutil.rmtree(log_dir, ignore_errors=True)
    train_writer, test_writer = getSummaryWriters(sess)
    
    # at the beginning previous time step data is set to zero 
    cycles = 200000
    testStep = 500
    saveStep = 1000
    log_step = 5
    create_buffer()
    _cur_state = np.zeros([SENSOR_DATA_DIM, STATE_SIZE], dtype=np.float32)
    # print(type(_cur_state))
    for i in range(cycles):
        truth_val = truth_values(1)
        sensed_data = [sense.allSensors_read()]
        _, hour_pred, detection_pred, head_angle, _cur_state,loss_val,summary = sess.run([
            optim_t,
            hour_t,
            detection_t,
            motor_t,
            cur_state,
            loss_tensor,
            merged
            ], 
            feed_dict={
                input_series: SENSE_BUFFER[0],
                truth_series: SENSE_BUFFER[1],
                init_state: _cur_state
            })
        
        updateBuffer()
        
        if i % log_step == 0: train_writer.add_summary(summary, i)
        if i % saveStep == 0: saveModel(sess, model_save_path[0])

        # print(hour_pred, detection_pred, head_angle, truth_val[0][:2], out[0].tolist())
        print('Time: %s, Detection: %s, Angle: %s'%(hour_pred,detection_pred,head_angle))
        print('Loss: %s'%loss_val)

        detection_LED.write(detection_pred)
        head.write(min(max(0,head_angle),180))
        time.sleep(0.5)
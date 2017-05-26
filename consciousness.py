from theBrain import *
from datetime import datetime

def truth_values(batch_size):
    # this is supposed to be high when I am here
    hour = datetime.now().hour/NUM_HOURS
    detection = int(sense.IAM_HERE_PIN.read())
    angle = 0

    return [[hour, detection, angle]]*batch_size
    

sensor_t, prev_step, truth_t = get_placeholders()
output, prev_inf = run_model(sensor_t, prev_step)
loss_t, optimStep = loss_optim(output, truth_t)
h_accuracy, d_accuracy, acc_total = accuracy(output, truth_t)

denormalized = getInterpretation(output)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # creating new summary writers and deleting old summary logs
    # shutil.rmtree(log_dir, ignore_errors=True)
    # train_writer, test_writer = getSummaryWriters(sess)
    
    # at the beginning previous time step data is set to zero 
    prev = np.zeros([1,HIDDEN_DIM_3])

    cycles = 12000
    testStep = 500
    saveStep = 1000
    log_step = 100
    for i in range(cycles):
        truth_val = truth_values(1)
        den, out, prev, _ = sess.run([denormalized, output, prev_inf, optimStep], feed_dict={
            sensor_t: [sense.allSensors_read()],
            prev_step: prev,
            truth_t: truth_val
        })
    
        hour_pred = den[0].tolist()[0][0]
        detection_pred = den[1].tolist()[0][0]
        head_angle = den[2].tolist()[0][0]

        # print(hour_pred, detection_pred, head_angle, truth_val[0][:2], out[0].tolist())
        print('Time: %s, Detection: %s, Angle: %s'%(hour_pred,detection_pred,head_angle))

        head.write(head_angle)
        time.sleep(0.5)
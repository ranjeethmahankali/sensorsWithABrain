from sensing import *
from theBrain import *

temp = sensor("temp", 0)
light = sensor("light",1)
dist = sensor("distance", 2)
# sound
head = motor("head", 9)

sensor_t, truth_t = get_placeholders()
output = response(sensor_t)
loss_t, optimStep = loss_optim(output, truth_t)
answer, acc_val = answer_accuracy(output, truth_t)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # creating new summary writers and deleting old summary logs
    shutil.rmtree(log_dir, ignore_errors=True)
    train_writer, test_writer = getSummaryWriters(sess)

    cycles = 1200000
    testStep = 500
    saveStep = 4000
    log_step = 100
    startTime = time.time()
    test_batch_size = 700

    try:
        for i in range(cycles):
            batch = data.next_batch(batch_size)
            _, summary = sess.run([optim, merged], feed_dict={
                bottleneck: batch[0],
                target: batch[1],
                keep_prob:0.6
            })
            if i % log_step == 0: train_writer.add_summary(summary, i)

            timer = estimate_time(startTime, cycles, i)
            pL = 20 # this is the length of the progress bar to be displayed
            pNum = i % pL
            pBar = '#'*pNum + ' '*(pL - pNum)

            sys.stdout.write('...Training...|%s|-(%s/%s)- %s\r'%(pBar, i, cycles, timer))

            if i % testStep == 0:
                testBatch = data.test_batch(test_batch_size)
                summary, acc, lval, graph_out, vec = sess.run([
                    merged, 
                    accuracy, 
                    lossVal, 
                    graph, 
                    vector
                    ],feed_dict={
                        bottleneck: testBatch[0],
                        target: testBatch[1],
                        keep_prob:1.0
                    }
                )
                test_writer.add_summary(summary, i)

                g_sum = int(np.sum(graph_out))
                t_sum = int(np.sum(testBatch[1]))
                # tracker helps to compare the data being printed to previous run with same 
                # training examples
                # tracker = (i/testStep)%(2500/test_batch_size)
                # print(testBatch[0][0])
                # print(vec[0], testBatch[1][0])
                print('Acc: %.2f; L: %.2f; Sums: %s/%s%s'%(acc, lval,g_sum,t_sum,' '*40))
        
        # now saving the trained model every 1500 cycles
            if i % saveStep == 0 and i != 0:
                saveModel(sess, model_save_path[1])
        
        # saving the model in the end
        saveModel(sess, model_save_path[1])
    # if the training is interrupted from keyboard (ctrl + c)
    except KeyboardInterrupt:
        print('')
        print('You interrupted the training process')
        decision = input('Do you want to save the current model before exiting? (y/n):')

        if decision == 'y':
            saveModel(sess, model_save_path[1])
        elif decision == 'n':
            print('\n...Model not saved...')
            pass
        else:
            print('\n...Invalid input. Model not saved...')
            pass
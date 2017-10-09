'''
code snippet to run a tensorflow session for performing the training.
'''

'''
    Update: This now includes a functional interface to the session runner which is why, no global graph is required.
    I will make it even more isolated from the external dependencies.
'''

''' The execute_graph function is used for training the model '''
# function to execute the session and train the model:
def execute_graph(dataX, dataY, exec_graph, model_name, no_of_iterations):
    '''
        function to start and execute the session with training.
        @param
        dataX, dataY => the data to train on
        exec_graph => the computation graph to be trained
        model_name => the name of the model where the files will be saved
        no_of_itreations => no of iterations for which the model needs to be trained
        @return => Nothing, this function has a side effect
    '''
    assert dataX.shape[-1] == dataY.shape[-1], "The Dimensions of input X and labels Y don't match"

    # the number of examples in the dataset
    no_of_examples = dataX.shape[-1]

    with tf.Session(graph=exec_graph) as sess:
        # create the tensorboard writer for collecting summaries:
        log_dir = os.path.join(base_model_path, model_name)
        tensorboard_writer = tf.summary.FileWriter(logdir=log_dir, graph=sess.graph, filename_suffix=".bot")

        # The saver object for saving and loading the model
        saver = tf.train.Saver(max_to_keep=2)

        # check if the model has been saved.
        model_path = log_dir
        model_file = os.path.join(model_path, model_name) # the name of the model is same as dir
        if(os.path.isfile(os.path.join(base_model_path, model_name, "checkpoint"))):
            # the model exists and you can restore the weights
            saver.restore(sess, tf.train.latest_checkpoint(model_path))
        else:
            # no saved model found. so, run the global variables initializer:
            sess.run(init_op)

        print "Starting the training ..."
        print "==============================================================================================="

        batch_index = 0 # initialize it to 0
        # start the training:
        for iteration in range(no_of_itreations):

            # fetch the input and create the batch:
            start = batch_index; end = start + batch_size
            inp_X = dataX[:, start: end].T # extract the input features
            inp_Y = dataY[:, start: end].T # extract the labels

            # feed the input to the graph and get the output:
            _, cost = sess.run((train_step, loss), feed_dict={input_X: inp_X, labels_Y: inp_Y})

            # checkpoint the model at certain times
            if((iteration + 1) % checkpoint_factor == 0):
                # compute the summary:
                summary = sess.run(all_summaries, feed_dict={input_X: inp_X, labels_Y: inp_Y})

                # accumulate the summary
                tensorboard_writer.add_summary(summary, (iteration + 1))

                # print the cost at this point
                print "Iteration: " + str(iteration + 1) + " Current cost: " + str(cost)

                # save the model trained so far:
                saver.save(sess, model_file, global_step = (iteration + 1))

            # increment the batch_index
            batch_index = (batch_index + batch_size) % no_of_examples

        print "==============================================================================================="
        print "Training complete"


    ''' calc_accuracy function can internally use generate_predictions function to obtain the predictions.
        But I have kept them separate for less dependencies. '''
    ''' Function for calculating the Accuracy of the classifier: '''
    def calc_accuracy(dataX, dataY, exec_graph, model_name, threshold = 0.5):
    '''
        Function to run the trained model and calculate it's accuracy on the given inputs
        @param
        dataX, dataY => The data to be used for accuracy calculation
        exec_graph => the Computation graph to be used
        model_name => the model to restore the weights from
        threshold => the accuracy threshold (by default it is 0.5)
        @return => None (function has side effect)
    '''
    assert dataX.shape[-1] == dataY.shape[-1], "The Dimensions of input X and labels Y don't match"

    # the number of examples in the dataset
    no_of_examples = dataX.shape[-1]

    with tf.Session(graph=exec_graph) as sess:

        # The saver object for saving and loading the model
        saver = tf.train.Saver(max_to_keep=2)

        # the model must exist and you must be able to restore the weights
        model_path = os.path.join(base_model_path, model_name)
        assert os.path.isfile(os.path.join(model_path, "checkpoint")), "Model doesn't exist"

        saver.restore(sess, tf.train.latest_checkpoint(model_path))

        # compute the predictions given out by model
        preds = sess.run(prediction, feed_dict={input_X: dataX.T, labels_Y: dataY.T})

        encoded_preds = (preds >= threshold).astype(np.float32)

        # calculate the accuracy in percentage:
        correct = np.sum((encoded_preds == dataY.T).astype(np.int32))
        accuracy = (float(correct) / dataX.shape[-1]) * 100 # for percentage

    # return the so calculated accuracy:
    return accuracy



    ''' Function for providing the predictions using the trained model '''
    # this function will be quite similar to the accuracy calculation function.
    def generate_predictions(dataX, exec_graph, model_name):
    '''
        Function to run the trained model and generate predictions for the given data
        @param
        dataX => The data to be used for accuracy calculation
        exec_graph => the Computation graph to be used
        model_name => the model to restore the weights from
        @return => predictions array returned
    '''

    # the number of examples in the dataset
    no_of_examples = dataX.shape[-1]

    with tf.Session(graph=exec_graph) as sess:

        # The saver object for saving and loading the model
        saver = tf.train.Saver(max_to_keep=2)

        # the model must exist and you must be able to restore the weights
        model_path = os.path.join(base_model_path, model_name)
        assert os.path.isfile(os.path.join(model_path, "checkpoint")), "Model doesn't exist"

        saver.restore(sess, tf.train.latest_checkpoint(model_path))

        # compute the predictions given out by model
        preds = sess.run(prediction, feed_dict={input_X: dataX.T})

    # return the so calculated accuracy:
    return preds

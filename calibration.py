import tensorflow as tf


def temperature_scale(logits, session, valid_labels, learning_rate=0.01, num_steps=50):
    """
    Calibrate the confidence prediction using temperature scaling.
    :param logits:          Outputs of the neural network before softmax.
    :param session:         Tensorflow session.
    :param x_pl:            Placeholder for the inputs to the NN.
    :param y_pl:            Placeholder for the label for the loss.
    :param valid_data:      Validation inputs.
    :param valid_labels:    Validation labels.
    :return:                Scaled predictions op.
    """

    temperature = tf.Variable(initial_value=1., trainable=True, dtype=tf.float32, name="temperature")
    scaled_logits = logits / temperature

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=valid_labels, logits=scaled_logits))

    opt = tf.train.MomentumOptimizer(learning_rate, 0.9)
    opt_step = opt.minimize(loss, var_list=[temperature])

    session.run(tf.variables_initializer([temperature]))
    session.run(tf.variables_initializer([opt.get_slot(var, name) for name in opt.get_slot_names() for var in [temperature]]))

    for i in range(num_steps):
        session.run(opt_step)

    return temperature

import tensorflow as tf
from tensorflow import keras
from utill import Makedir
"""
# learning rate function
def step_decay(epoch):
   initial_lrate = 1e-03
   drop = 0.1
   epochs_drop = 20.0
   lrate = initial_lrate * math.pow(drop,
                                    math.floor((1+epoch)/epochs_drop))
   return lrate
"""

def model_fit(model, train_gen, train_steps, epochs, val_gen, val_steps, class_num, PATH, model_name):
    # define loss and optimizer
    save_path = Makedir(PATH, 'weight')
    save_path = save_path + "/" + model_name + '_{}'

    optimizer = keras.optimizers.Adam(learning_rate=1e-3)
    if class_num == 2:
        loss_func = keras.losses.BinaryCrossentropy()
        train_loss = tf.keras.metrics.Mean(name='train_loss')
        train_accuracy = tf.keras.metrics.BinaryAccuracy(name='train_accuracy')
        valid_loss = tf.keras.metrics.Mean(name='valid_loss')
        valid_accuracy = tf.keras.metrics.BinaryAccuracy(name='valid_accuracy')
    else:
        loss_func = keras.losses.CategoricalCrossentropy()
        train_loss = tf.keras.metrics.Mean(name='train_loss')
        train_accuracy = tf.keras.metrics.CategoricalAccuracy(
            name='train_accuracy')

        valid_loss = tf.keras.metrics.Mean(name='valid_loss')
        valid_accuracy = tf.keras.metrics.CategoricalAccuracy(
            name='valid_accuracy')

    train_losses = []
    train_acces = []
    val_losses = []
    val_acces = []

    @tf.function
    def train_step(images, labels):
        with tf.GradientTape() as tape:
            predictions = model(images, training=True)
            loss = loss_func(y_true=labels, y_pred=predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(grads_and_vars=zip(
            gradients, model.trainable_variables))

        train_loss(loss)
        train_accuracy(labels, predictions)

    @tf.function
    def valid_step(images, labels):

        predictions = model(images, training=False)
        v_loss = loss_func(labels, predictions)

        valid_loss(v_loss)
        valid_accuracy(labels, predictions)

    # start training
    for epoch in range(epochs):
        train_loss.reset_states()
        train_accuracy.reset_states()
        valid_loss.reset_states()
        valid_accuracy.reset_states()
        #optimizer.learning_rate = step_decay(epoch)

        #step = 0
        for step in range(train_steps):
            print('.', end='')
            if (step+1) % 100 == 0:
                print()
            images, labels = train_gen()
            train_step(images, labels)

        print('x', 'current_learning_rate : {}'.format(
            optimizer.learning_rate), sep=' ')

        for val_step in range(val_steps):
            if (val_step+1) % 100 == 0:
                print('.', end='')  
            valid_images, valid_labels = val_gen()
            valid_step(valid_images, valid_labels)
        print('x')
        print("Epoch: {}/{}, train loss: {:.5f}, train accuracy: {:.5f}, "
              "valid loss: {:.5f}, valid accuracy: {:.5f}".format(epoch + 1,
                                                                  epochs,
                                                                  train_loss.result(),
                                                                  train_accuracy.result(),
                                                                  valid_loss.result(),
                                                                  valid_accuracy.result()))
        train_losses.append(train_loss.result())
        train_acces.append(train_accuracy.result())
        val_losses.append(valid_loss.result())
        val_acces.append(valid_accuracy.result())

        #if (epoch+1) % 10 == 0:
        #    model.save_weights(
        #        '/content/gdrive/My Drive/save_model/efficientnet/efficientnet_{}'.format(epoch+1+100), save_format='tf')
        
        if (epoch+1) % 10 == 0:
            model.save_weights(
                save_path.format(epoch+1), save_format='tf')

    history = {'train_losses': train_losses, 'train_acces': train_acces,
               'val_losses': val_losses, 'val_acces': val_acces}
    return history

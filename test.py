import tensorflow as tf
from tensorflow import keras

def model_test(model, test_data_gen, test_steps):
    loss_func = keras.losses.CategoricalCrossentropy()
    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.CategoricalAccuracy(
        name='test_accuracy')

    @tf.function
    def test_step(images, labels):
        predictions = model(images, training=False)
        t_loss = loss_func(labels, predictions)

        test_loss(t_loss)
        test_accuracy(labels, predictions)

    for step in range(test_steps):
        if step % 100 == 0:
            print('.', end='')
        test_images, test_labels = test_data_gen()
        test_step(test_images, test_labels)

    print('x')
    print("loss: {:.5f}, test accuracy: {:.5f}".format(test_loss.result(),
                                                       test_accuracy.result()))

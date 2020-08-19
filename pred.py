import numpy as np
import tensorflow as tf
from tensorflow import keras

def model_pred(model, test_data_gen, test_steps):
    predictions = []
    labels = []

    @tf.function
    def pred_step(images):
        prediction = model(images, training=False)
        return prediction

    for step in range(test_steps):
        if step % 100 == 0:
            print('.', end='')
        test_images, label = test_data_gen()
        prediction = pred_step(test_images)
        labels.append(label)        
        predictions.append(prediction)
    print('x')

    re_dic = dict({'pred':predictions, 'labels':labels})

    return re_dic

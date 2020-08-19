import tensorflow as tf
import matplotlib.pyplot as plt
import os
import numpy as np
import cv2

"""
수정사항
1. 이미지 저장시 예측 이름으로 저장 --> 이미지의 클래스 명이 있는 리스트/튜플 매개변수로 필요
"""

def Makedir(PATH, new_folder):
    PATH = os.path.join(PATH, new_folder)
    try:
        if not os.path.exists(PATH):
            os.makedirs(PATH)
    except OSError:
        print('Error Creating director')
    return PATH

def grad_cam_test2(model, seq_num, image_paths, save_path, class_names, IMAGE_SHAPE):
  save_path_root_directory = Makedir(save_path, 'grad_cam')
  seq_name = "seq"+str(seq_num + 1)
  layer_name = "seq_" + str(seq_num + 1)
  # make sequential directory 
  save_path_seq_directory = Makedir(save_path_root_directory, seq_name)

  a = 0

  # grad_cam start
  for path in image_paths:
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = cv2.resize(image, (IMAGE_SHAPE[1], IMAGE_SHAPE[0]))
    x = img.copy()
    x.astype(np.float32)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0

    with tf.GradientTape() as tape:
        inputs = tf.cast(x, tf.float32)
        seq_outputs, predictions = model(inputs, cam = 'grad')
        loss = predictions[:,0]

    seq_outputs = seq_outputs[seq_num]
    grads = tape.gradient(loss, seq_outputs)

    guided_grads = (
          tf.cast(seq_outputs > 0, "float32") * tf.cast(grads > 0, "float32") * grads
    )

    prediction = predictions[0][1]
    seq_outputs = seq_outputs[0]
    
    weights = np.mean(grads, axis=(1, 2))
    weights = weights.reshape(-1, 1)

    cam = (prediction - 0.5) * np.matmul(seq_outputs, weights)
    cam -= np.min(cam)
    cam /= np.max(cam)
    cam -= 0.2
    cam /= 0.8

    try:
      cam = cv2.resize(np.float32(cam), (IMAGE_SHAPE[1], IMAGE_SHAPE[0]))
    except Exception as e:
      #print(cam.shape)
      print(str(e))


    heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
    # heatmap[np.where(cam <= 0.2)] = 0
    grad_cam = cv2.addWeighted(img, 0.8, heatmap, 0.4, 0)
    save_cam = grad_cam[:, :, ::-1]
    a += 1
    # 이미지 예측값 image_save_path에 넣어서 저장!!!!!!!!!!
    #image_save_path = save_path_directory + '/grad_cam_' + layer_name + "_" + class_names[예측값] + "_" + 예측값 + '.png'
    image_save_path = save_path_seq_directory + '/grad_cam_'+layer_name+'_'+str(a)+'.png'
    cv2.imwrite(image_save_path, save_cam)

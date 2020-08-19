import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import os

# mkdir function
def Makedir(PATH, new_folder):
    PATH = os.path.join(PATH, new_folder)
    try:
        if not os.path.exists(PATH):
            os.makedirs(PATH)
    except OSError:
        print('Error Creating director')
    return PATH

def select_seq(seq_name):
    num_seq = int(seq_name[-1])-1
    return num_seq    

# Total data table function -- 데이터 폴더 보고 다시 작성
def check_images(labels, classes, path):
    save_path = Makedir(path, 'check_images')
    save_path = save_path + '/data_distribution.png'
    healthy = labels.healthy
    multiple_diseases = labels.multiple_diseases
    rust = labels.rust
    scab = labels.scab

    num_healthy = [len([i for i in healthy if i == 1]),
                    len([i for i in healthy if i == 0])]
    num_multiple = [len([i for i in multiple_diseases if i == 1]), len(
        [i for i in multiple_diseases if i == 0])]
    num_rust = [len([i for i in rust if i == 1]),
                len([i for i in rust if i == 0])]
    num_scab = [len([i for i in scab if i == 1]),
                len([i for i in scab if i == 0])]

    pos = [num_healthy[0], num_multiple[0], num_rust[0], num_scab[0]]
    nev = [num_healthy[1], num_multiple[1], num_rust[1], num_scab[1]]

    plt.rcParams["font.size"] = 12

    plt.figure(figsize=(12, 8))

    x = np.arange(len(classes))
    p1 = plt.bar(x-0.15, pos, width=0.3, color='#FF0000', label=1, alpha=0.5)
    plt.xticks(x, classes)
    p2 = plt.bar(x+0.15, nev, width=0.3, color='#0000FF', label=0, alpha=0.5)
    plt.xticks(x, classes)

    for i, rect in enumerate(p1):
        plt.text(rect.get_x() + rect.get_width() / 2.0, 0.95 *
                rect.get_height(), str(pos[i]), ha='center')
    for i, rect in enumerate(p2):
        plt.text(rect.get_x() + rect.get_width() / 2.0, 0.95 *
                rect.get_height(), str(nev[i]), ha='center')
    plt.legend((p1[0], p2[0]), ('1', '0'), fontsize=15)

    plt.savefig(save_path)
    plt.close()

# draw histogram graph function
def his_graph(history, epoch, path):
    save_path = Makedir(path, 'history')
    save_path = save_path + '/history.png'
    acc = history['train_acces']
    val_acc = history['val_acces']

    loss = history['train_losses']
    val_loss = history['val_losses']

    epochs_range = range(epoch)

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')

    plt.savefig(save_path)
    plt.close()

# confusion matrix function
def pred_confusion_matrix(pred, labels, class_num, path):
    save_path = Makedir(path, 'confusion_matrix')
    save_path = save_path + '/confusion_matrix.png'
    pred_int = []
    label_int = []

    # binary part 우선 제외
    # if class_num == 2:
    #     for i in range(len(pred)):
    #         if pred[i] > 0.5:
    #             pred_int.append(1)
    #         elif pred[i] <= 0.5:
    #             pred_int.append(0)
    #     label_int = labels
    # else :
    #multi-class part
    for j in range(len(pred)):
        max_value = max(pred[j][0])
        for u in range(class_num):
            index_value = pred[j][0][u]
            if max_value == index_value:
                pred_int.append(u)
            if labels[j][0][u] == 1:
                label_int.append(u)

    plt.title('Confusion Matrix')
    conf_matrix = confusion_matrix(label_int, pred_int)
    # 표 형식으로만 출력(필요하면 별도로 사용).
    sns.heatmap(conf_matrix, cmap="Blues", annot=True, fmt='g')
    plt.xlabel('predicted value')
    plt.ylabel('true value')
    plt.savefig(save_path)
    plt.close()


# draw feature map function
def Layers_predict(model, image):
    output_layers = [layer.output for layer in model.layers[:-2]]
    output_names = [layer.name for layer in model.layers[:-2]]
    feature_model = tf.keras.models.Model(inputs=model.input, outputs=output_layers)
    features = feature_model.predict(image)
    return features, output_names


def show_predict_image(show_model_pred, output_names, path, idx):
    save_path = Makedir(path, 'feature_map')
    save_path = save_path + '/feature_{}'.format(idx) + '.png'
    n_col = 16
    _, _, size, n_features = show_model_pred.shape
    n_row = n_features // n_col
    feature_map_image = np.zeros(
        shape=(size*n_row, size*n_col), dtype=('uint8'))
    for row in range(n_row):
        for col in range(n_col):
            input_fmi = show_model_pred[0, :, :, row*n_col+col]

            input_fmi -= input_fmi.mean()
            input_fmi /= input_fmi.std()
            input_fmi *= 64
            input_fmi += 128
            input_fmi = np.clip(input_fmi, 0, 255).astype('uint8')

            feature_map_image[row*size:(row+1)*size,
                              col*size:(col+1)*size] = input_fmi

    plt.figure(figsize=(n_col, n_row))
    plt.xticks([])
    plt.yticks([])
    plt.title('layer : {}'.format(output_names))
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def get_feature(model, image_path, result_path, target_size):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image/255.0
    image = cv2.resize(image, (target_size[0], target_size[1]))

    layer_predict, layer_names = Layers_predict(model, image)
    for i in range(len(layer_predict)):
        show_predict_image(layer_predict[i], layer_names[i], result_path, i)

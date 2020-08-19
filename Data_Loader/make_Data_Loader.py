import tensorflow as tf
import pandas as pd
import numpy as np

import os
import cv2
import albumentations as A

from sklearn.model_selection import train_test_split

class Classification_Dataset:
    # class_list -> image_labels 변경
    def __init__(self, image_paths = "", one_hot_label = None, integers_label = None, target_size = None, augment = None):
        
        self.image_list = image_paths
        self.one_hot_label = one_hot_label
        self.integers_label = integers_label
        self.target_size = target_size
        self.augment = augment
    
    def __len__(self):
        if self.one_hot_label:
            return len(self.image_list)
        elif self.integers_label:
            return len(self.integers_label)
    
    def __getitem__(self, idx):
        image = cv2.imread(self.image_list[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image/255.0

        if self.one_hot_label:
            label = self.one_hot_label[idx]
        
        if self.integers_label:
            label = 0

            for i in range(len(self.integers_label)):
                if(self.integers_label[i] in self.image_list[idx]):
                    label = i
                    break             

        
        if self.augment:
            transformed = self.augment(image=image)
            image = transformed['image']

        #albumentations module 따로 설정함.
        #image = image.resize(self.target_size)
        return image, label



class Classification_Data_Loader(tf.keras.Sequential):
    def __init__(self, dataset, batch_size=None, shuffle=False):        
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle=shuffle
        self.index_list = [idx for idx in range(len(self.dataset))]
        self.idx=0
        
    def __getitem__(self, idx):      
        start = idx * self.batch_size
        end = (idx+1) * self.batch_size
        data = []
        label = []
        
        if self.shuffle:
            np.random.shuffle(self.index_list)
            
        for j in range(start,end):
            if j >= len(self.index_list):
                j%=len(self.dataset)
            data.append(self.dataset[self.index_list[j]])
      
        batch = tuple(tf.stack(sample, axis=0) for sample in zip(*data))

        if self.idx >= (len(self.dataset)//self.batch_size):
            self.idx=0
        self.idx +=1
        return batch

    def __call__(self):
        batch = self.__getitem__(self.idx)
        return batch

    def __len__(self):
        return (len(self.dataset) // self.batch_size)

    def get_batch(self):
        return self.batch_size



def make_Data_Loader(IMAGE_PATH, IMAGE_SHAPE, train_batch, valid_batch, test_batch):
    #train, test, sample csv 읽어오기
    train = pd.read_csv('/content/drive/My Drive/cvpr_data/train.csv', index_col=None)
    test = pd.read_csv('/content/drive/My Drive/cvpr_data/test.csv', index_col=None)
    sample = pd.read_csv('/content/drive/My Drive/cvpr_data/sample_submission.csv', index_col=None)


    images = []
    test_images = []

    image_ids = list(train.image_id)
    test_ids = list(test.image_id)

    for image_id in image_ids:
      images.append(os.path.join(IMAGE_PATH,image_id) + '.jpg')

    for test_id in test_ids:
      test_images.append(os.path.join(IMAGE_PATH, test_id) + '.jpg')

    labels = list(zip(train.healthy, train.multiple_diseases, train.rust, train.scab))
    test_labels = []

    train_images, val_images, train_labels, val_labels = train_test_split(images, labels, test_size = 0.2, random_state=2020, stratify=labels)

    #albumentations Module 설정.
    transforms_train = A.Compose([
        A.Resize(height=IMAGE_SHAPE[0], width=IMAGE_SHAPE[1], p=1.0),
        A.RandomCrop(height=IMAGE_SHAPE[0]-2, width=IMAGE_SHAPE[1]-2, p=1.0),
        A.Resize(height=IMAGE_SHAPE[0], width=IMAGE_SHAPE[1], p=1.0),
        A.Flip(),
        A.ShiftScaleRotate(rotate_limit=40.0, p=0.8),
        A.HorizontalFlip(p=0.5)
    ])

    transforms_valid = A.Compose([
        A.Resize(height=IMAGE_SHAPE[0], width=IMAGE_SHAPE[1], p=1.0)
    ])

    transforms_test = A.Compose([
        A.Resize(height=IMAGE_SHAPE[0], width=IMAGE_SHAPE[1], p=1.0)
    ])

    train_dataset = Classification_Dataset(image_paths=train_images, one_hot_label = train_labels, augment = transforms_train)
    valid_dataset = Classification_Dataset(image_paths=val_images, one_hot_label = val_labels, augment = transforms_valid)
    test_dataset = Classification_Dataset(image_paths=images, one_hot_label = labels, augment = transforms_test)

    train_data_gen = Classification_Data_Loader(dataset=train_dataset, batch_size=train_batch, shuffle=True)
    valid_data_gen = Classification_Data_Loader(dataset=valid_dataset, batch_size=valid_batch, shuffle=True)
    test_data_gen = Classification_Data_Loader(dataset=test_dataset, batch_size=test_batch)
    
    return train_data_gen, valid_data_gen, test_data_gen

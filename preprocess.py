from PIL import Image
import numpy as np
import cv2
import random
from scipy.ndimage import zoom


zoom_var = [0.8,0.9,1.1,1.2,1.3]

def clipped_zoom(img, zoom_factor, **kwargs):

    h, w = img.shape[:2]
        #print(h,w)

        # For multichannel images we don't want to apply the zoom factor to the RGB
        # dimension, so instead we create a tuple of zoom factors, one per array
        # dimension, with 1's for any trailing dimensions after the width and height.
    zoom_tuple = (zoom_factor,) * 2 + (1,) * (img.ndim - 2)

        # Zooming out
    if zoom_factor < 1:

            # Bounding box of the zoomed-out image within the output array
        zh = int(np.round(h * zoom_factor))
        zw = int(np.round(w * zoom_factor))
        top = (h - zh) // 2
        left = (w - zw) // 2

            # Zero-padding
        out = np.zeros_like(img)
        out[top:top+zh, left:left+zw] = zoom(img, zoom_tuple, **kwargs)
            #print(out.shape)
        # Zooming in
    elif zoom_factor > 1:

            # Bounding box of the zoomed-in region within the input array
        zh = int(np.round(h / zoom_factor))
        zw = int(np.round(w / zoom_factor))
        top = (h - zh) // 2
        left = (w - zw) // 2

        out = zoom(img[top:top+zh, left:left+zw], zoom_tuple, **kwargs)

            # `out` might still be slightly larger than `img` due to rounding, so
            # trim off any extra pixels at the edges
        trim_top = ((out.shape[0] - h) // 2)
        trim_left = ((out.shape[1] - w) // 2)
        out = out[trim_top:trim_top+h, trim_left:trim_left+w]
            #print(out.shape)

        # If zoom_factor == 1, just return the input array
    else:
        out = img
    return out



def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")

    return cv2.LUT(image, table)


class Preprocessing:
    def __init__(self,task = 'ceph'):
        
        self.task = task
        #self.standard = standard
        # self.rotation_range = rotation_range

        #self.windowing_min = wlevel - wwidth//2
        #self.windowing_max = wlevel + wwidth//2
    
    def _nomalized(self, image, mask):
        image = image.astype('float32')
        image /= 255
        
        mask = mask.astype('float32')
        mask /= 255
        #mask[mask <= 0.5] = 0.
        #mask[mask > 0.5] = 1
        
        return image, mask
    
    def _rotation_2D(self, img, mask, degree = 10):
        p = 0.5
        R_move = random.randint(-degree,degree)
        probability = random.random()
        if probability < p:
            M = cv2.getRotationMatrix2D((img.shape[1]/2, img.shape[0]/2), R_move, 1)
            img = cv2.warpAffine(img,M,(img.shape[1],img.shape[0]))
            mask = cv2.warpAffine(mask,M,(img.shape[1],img.shape[0]))
            #rotate_pimg = cv2.warpAffine(point_img,M,(img.shape[0],img.shape[1]))
            
        return img, mask
    
    def _shift_2D(self, img, mask, shift = 30):
        p = 0.5
        x_move = random.randint(-shift,shift)
        y_move = random.randint(-shift,shift)
        if random.random() < p:
            #print("_shift_2D")
            shift_M = np.float32([[1,0,x_move], [0,1,y_move]])
            img = cv2.warpAffine(img, shift_M,(img.shape[1], img.shape[0]))
            mask = cv2.warpAffine(mask, shift_M, (mask.shape[1], mask.shape[0]))
            #rotate_pimg = cv2.warpAffine(point_img,M,(img.shape[0],img.shape[1]))
            
        return img, mask
    
    def _blur_2D(self, img):
        p = 0.5
        if random.random() < p:
            #print("_blur_2D")
            img = cv2.blur(img,(5,5))
            #rotate_pimg = cv2.warpAffine(point_img,M,(img.shape[0],img.shape[1]))
            
        return img
     
    def _sharpning_2D(self, img):
        p = 0.5
        if random.random() < p:
            #print("_sharpning_2D")
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            img = cv2.filter2D(img, -1, kernel)
            #rotate_pimg = cv2.warpAffine(point_img,M,(img.shape[0],img.shape[1]))
            
        return img
    
    def _gamma_2D(self, img):
        p = 0.5
        if random.random() < p:
            numlist = [0.5,0.8,1.1,1.5,1.8,2.0,2.3,2.6,2.9,3.2,3.5]
            var_gamma = random.sample(numlist, 1)
            var_gamma = var_gamma[0]
            img = adjust_gamma(img, gamma=var_gamma)
            
        return img
               
    def _zoom_2D(self, img, mask):
        p = 0.5
        if random.random() < p:
            #print('_zoom_2D')
            zoom_var_list = random.randint(0,4)
                           
            img = clipped_zoom(img, zoom_var[zoom_var_list])
            mask = clipped_zoom(mask, zoom_var[zoom_var_list])
            #rotate_pimg = cv2.warpAffine(point_img,M,(img.shape[0],img.shape[1]))
            
        return img, mask
                          
    def clipped_zoom(img, zoom_factor, **kwargs):

        h, w = img.shape[:2]
        #print(h,w)

        # For multichannel images we don't want to apply the zoom factor to the RGB
        # dimension, so instead we create a tuple of zoom factors, one per array
        # dimension, with 1's for any trailing dimensions after the width and height.
        zoom_tuple = (zoom_factor,) * 2 + (1,) * (img.ndim - 2)

        # Zooming out
        if zoom_factor < 1:

            # Bounding box of the zoomed-out image within the output array
            zh = int(np.round(h * zoom_factor))
            zw = int(np.round(w * zoom_factor))
            top = (h - zh) // 2
            left = (w - zw) // 2

            # Zero-padding
            out = np.zeros_like(img)
            out[top:top+zh, left:left+zw] = zoom(img, zoom_tuple, **kwargs)
            #print(out.shape)
        # Zooming in
        elif zoom_factor > 1:

            # Bounding box of the zoomed-in region within the input array
            zh = int(np.round(h / zoom_factor))
            zw = int(np.round(w / zoom_factor))
            top = (h - zh) // 2
            left = (w - zw) // 2

            out = zoom(img[top:top+zh, left:left+zw], zoom_tuple, **kwargs)

            # `out` might still be slightly larger than `img` due to rounding, so
            # trim off any extra pixels at the edges
            trim_top = ((out.shape[0] - h) // 2)
            trim_left = ((out.shape[1] - w) // 2)
            out = out[trim_top:trim_top+h, trim_left:trim_left+w]
            #print(out.shape)

        # If zoom_factor == 1, just return the input array
        else:
            out = img
        return out
    """
    def _array2img(self, x):
        x = sitk.GetArrayFromImage(x).astype('float32')
        return x
    def _getvoi(self, xx):
        img, voi = xx
        if self.task in ['all', 'ascites']:
            img = img[voi[0]:voi[1],:,:]
        elif self.task == 'varix':
            img = img[voi[0]:voi[1],voi[2]:voi[3],voi[4]:voi[5]]
        
        return img
    def _resize(self, x):
        for i in range(x.shape[0]):
            temp = ndimage.zoom(x[i], [.5, .5], order=0, mode='constant', cval=0.)
            if i == 0:
                result = np.zeros((x.shape[0],)+temp.shape)
            result[i] = temp
        return result
    def _windowing(self, x):
        x = np.clip(x, self.windowing_min, self.windowing_max)
        return x
    def _standard(self, x):
        if self.standard == 'minmax':
            x = (x - self.windowing_min) / (self.windowing_max - self.windowing_min)
        elif self.standard == 'normal':
            x = (x - x.mean()) / x.std()
        else:
            pass
        return x
    def _onehot(self, x):
        result = np.zeros((2,))
        if x == 0:
            result[0] = 1
        else:
            result[1] = 1
        return result[np.newaxis,...]
    def _expand(self, x):
        x = x[np.newaxis,...,np.newaxis]
        return x
    def _rotation(self, x, theta=None, dep_index=0, row_index=1, col_index=2, fill_mode='nearest', cval=0.):
        if theta:
            theta1, theta2, theta3 = theta
        else:
            theta1 = np.pi / 180 * np.random.uniform(-self.rotation_range[0], self.rotation_range[0])
            theta2 = np.pi / 180 * np.random.uniform(-self.rotation_range[1], self.rotation_range[1])
            theta3 = np.pi / 180 * np.random.uniform(-self.rotation_range[2], self.rotation_range[2])
        rotation_matrix_z = np.array([[np.cos(theta1), -np.sin(theta1), 0, 0],
                                      [np.sin(theta1), np.cos(theta1), 0, 0],
                                      [0, 0, 1, 0],
                                      [0, 0, 0, 1]])
        rotation_matrix_y = np.array([[np.cos(theta2), 0, -np.sin(theta2), 0],
                                      [0, 1, 0, 0],
                                      [np.sin(theta2), 0, np.cos(theta2), 0],
                                      [0, 0, 0, 1]])
        rotation_matrix_x = np.array([[1, 0, 0, 0],
                                      [0, np.cos(theta3), -np.sin(theta3), 0],
                                      [0, np.sin(theta3), np.cos(theta3), 0],
                                      [0, 0, 0, 1]])
        rotation_matrix = np.dot(np.dot(rotation_matrix_y, rotation_matrix_z), rotation_matrix_x)
        d, h, w = x.shape[dep_index], x.shape[row_index], x.shape[col_index]
        transform_matrix = self.__transform_matrix_offset_center(rotation_matrix, d, w, h)
        x = self.__apply_transform(x, transform_matrix, fill_mode, cval)
        return x
    def __transform_matrix_offset_center(self, matrix, x, y, z):
        o_x = float(x) / 2 + 0.5
        o_y = float(y) / 2 + 0.5
        o_z = float(z) / 2 + 0.5
        offset_matrix = np.array([[1, 0, 0, o_x],
                                  [0, 1, 0, o_y], 
                                  [0, 0, 1, o_z], 
                                  [0, 0, 0, 1]])
        reset_matrix = np.array([[1, 0, 0, -o_x], 
                                 [0, 1, 0, -o_y], 
                                 [0, 0, 1, -o_z], 
                                 [0, 0, 0, 1]])
        transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
        return transform_matrix
    def __apply_transform(self, x, transform_matrix, fill_mode='nearest', cval=0.):
        final_affine_matrix = transform_matrix[:3, :3]
        final_offset = transform_matrix[:3, 3]
        x = ndimage.interpolation.affine_transform(x, 
                                                   final_affine_matrix, 
                                                   final_offset, 
                                                   order=0, 
                                                   mode=fill_mode, 
                                                   cval=cval)
        return x
   
    """

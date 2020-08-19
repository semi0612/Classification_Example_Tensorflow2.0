import numpy as np
import cv2
from preprocess import *
import random as rand



def generator_all(img_paths, mask_paths, batch_size=16, landmark_num = 0,mask_size = 30 ,P_H = 200, P_W = 200, mode = 'train'):

    seed=42
    prep = Preprocessing()
    mode = mode
    
    def _preprocessing(img,mask ,prep, voi=None):
        if(mode == 'train'): 
            img, mask = prep._rotation_2D(img, mask,10)
            img, mask = prep._shift_2D(img, mask,100)
            img,mask = prep._zoom_2D(img,mask)
            img = prep._blur_2D(img)
            img = prep._sharpning_2D(img)
            img = prep._gamma_2D(img)
                      
        img, mask = prep._nomalized(img, mask)
        return img, mask
    
    
    random.seed(seed)
    
    while True:
        n_data = len(img_paths)
        indices = list(range(n_data))
        random_data_path = np.random.choice(indices, n_data, replace=False)

        start = 0
        end = 0   
     
        #데이터셋 루프 데이터셋의 모든 이미지를 돌면 탈출
        while(end < n_data):
            start = end
            ##
            if end >= n_data:
                end = n_data

            batch_imgs = []
            batch_masks = [] 
                
            count = start
            
            count_end = start + batch_size
            if(count_end > len(img_paths)):
                count_end = len(img_paths)
            while(count < count_end):               
                ##이미지 로드             
                img_path = img_paths[count]
                mask_path = mask_paths[count]
   
                temp_image = cv2.imread(img_path,0)                    
                point = np.load(mask_path)
        
                H = temp_image.shape[0]
                W = temp_image.shape[1]
              
                check_x = point[landmark_num][0]
                check_y = point[landmark_num][1]
                #######################################################################
                       
                SHIFT_X = random.randint(-mask_size, mask_size)
                SHIFT_Y = random.randint(-mask_size, mask_size)

                origin_lu_x = point[landmark_num][0] - int(P_W/2)
                origin_lu_y = point[landmark_num][1] - int(P_H/2)
                origin_rd_x = point[landmark_num][0] + int(P_W/2)
                origin_rd_y = point[landmark_num][1] + int(P_H/2)

                agu_lu_x = origin_lu_x + SHIFT_X
                agu_lu_y = origin_lu_y + SHIFT_Y
                agu_rd_x = origin_rd_x + SHIFT_X
                agu_rd_y = origin_rd_y + SHIFT_Y
                
                max_x = temp_image.shape[0]
                max_y = temp_image.shape[1]
                   
                mask = np.zeros((H,W),dtype=np.uint8)
                cv2.circle(mask,(int(point[landmark_num][0]),int(point[landmark_num][1])),30,(255),-1 )
         
                image = temp_image[agu_lu_y:agu_rd_y,agu_lu_x:agu_rd_x]
                mask = mask[agu_lu_y:agu_rd_y,agu_lu_x:agu_rd_x]

                image, mask = _preprocessing(img=image, mask = mask, prep=prep)            
                      
                batch_imgs.append(image)
                batch_masks.append(mask)              
                count += 1
                                        
            end += batch_size    
                                  
            batch_imgs = np.array(batch_imgs)
            batch_masks = np.array(batch_masks)
            #print(batch_imgs.shape)
                
            batch_imgs = batch_imgs.reshape(-1,P_H,P_W,1)
            batch_masks = batch_masks.reshape(-1,P_H,P_W,1)

            yield [batch_imgs, batch_masks]
            


            
def new_generator_all(img_paths, mask_paths, batch_size=16, landmark_num = 0 ,H = 256, W = 256,  mode = 'train'):

    seed=42
    def _preprocessing(img,mask ,prep, voi=None):
        if(mode == 'train'): 
            img, mask = prep._rotation_2D(img, mask,10)
            img, mask = prep._shift_2D(img, mask,100)
            img,mask = prep._zoom_2D(img,mask)
            img = prep._blur_2D(img)
            img = prep._sharpning_2D(img)
            img = prep._gamma_2D(img)
                
        img, mask = prep._nomalized(img, mask)
        return img, mask
    
    
    prep = Preprocessing()
    random.seed(seed)
    
    while True:
        n_data = len(img_paths)
        indices = list(range(n_data))
        
        #랜덤 인덱스 배열
        random_data_path = np.random.choice(indices, n_data, replace=False)

        start = 0
        end = 0
        
        while(end < n_data):  # n_batch >= n_data이면 반복문 탈출
            start = end
            end += batch_size
            if end >= n_data:
                end = n_data

            batch_imgs = []
            batch_masks = [] 
            
            for i in random_data_path[start: end]:                                    
                img_path = img_paths[i]
                mask_path = mask_paths[i]         

                temp_image = cv2.imread(img_path,0)
                point = np.load(mask_path)
                         
                pimage = np.zeros((temp_image.shape[0]+int(W/2),temp_image.shape[1]+int(W/2)),dtype=np.uint8)
                pimage[0:temp_image.shape[0],0:temp_image.shape[1]] = temp_image
                
                pmask = np.zeros((temp_image.shape[0]+int(W/2),temp_image.shape[1]+int(W/2)),dtype=np.uint8)
                cv2.circle(pmask,(int(point[landmark_num][0]),int(point[landmark_num][1])),30,(255),-1 )
                
                SHIFT_X = random.randint(-50, 50)
                SHIFT_Y = random.randint(-50, 50)
                
                origin_lu_x = point[landmark_num][0] - int(W/2)
                origin_lu_y = point[landmark_num][1] - int(H/2)
                origin_rd_x = point[landmark_num][0] + int(W/2)
                origin_rd_y = point[landmark_num][1] + int(H/2)
                
                #이동까지한 패치의 최종 좌표
                agu_lu_x = origin_lu_x + SHIFT_X
                agu_lu_y = origin_lu_y + SHIFT_Y
                agu_rd_x = origin_rd_x + SHIFT_X
                agu_rd_y = origin_rd_y + SHIFT_Y
                
                image = pimage[agu_lu_y:agu_rd_y,agu_lu_x:agu_rd_x]
                mask = pmask[agu_lu_y:agu_rd_y,agu_lu_x:agu_rd_x]
                
                image, mask = _preprocessing(img=image, mask = mask, prep=prep)
                batch_imgs.append(image)
                batch_masks.append(mask)
                
                
            batch_imgs = np.array(batch_imgs)
            batch_masks = np.array(batch_masks)
                
            batch_imgs = batch_imgs.reshape(-1,W,H,1)
            batch_masks = batch_masks.reshape(-1,W,H,1)
            

            yield [batch_imgs, batch_masks]

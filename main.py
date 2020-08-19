import tensorflow as tf
import model_list
from grad_cam import grad_cam_test2
from Data_Loader.make_Data_Loader import make_Data_Loader
from train import model_fit
from pred import model_pred
from test import model_test
from utill import Makedir, select_seq, check_images, his_graph, pred_confusion_matrix, get_feature



import argparse
import os

IMAGE_PATH = '/content/drive/My Drive/cvpr_data/images/' # 이미지 경로
IMAGE_SIZE = (512,512,3)
RESULT_PATH = '/content/drive/My Drive/imagenet_project' # 최종 저장 경로

# 특징맵 테스트 이미지 경로
FEATURE_IMAGE_PATH = '/content/drive/My Drive/cvpr_data/images/Test_1820.jpg'

# 클래스 개수, train, valid, test 배치와 스텝
class_num = 4
train_batch = 10
train_step = 80
valid_batch = 10
valid_step = 20
test_batch = 1
test_step = 10

#parser 선언부
parser = argparse.ArgumentParser(description='Training Classification')
parser.add_argument('model',type=str, help='Model Name')
parser.add_argument('epochs',type=int, help='Epoch Size')
parser.add_argument('select_seq',type=str,
                help="Select Grad CAM sequential\n" 
                      "vggnet : seq_1, seq_2, seq_3, seq_4, seq_5\n" 
                      "resnet : seq_1, seq_2, seq_3, seq_4, seq_5\n" 
                      "densenet : seq_1, seq_2, seq_3, seq_4\n" 
                      "efficientnet : seq_1, seq_2, seq_3, seq_4, seq_5, seq_6, seq_7, conv_2\n" 
                      "resnextnet : seq_1, seq_2, seq_3, seq_4")

#attention 옵션
parser.add_argument("--attention",choices=['se','cbam'])

args = parser.parse_args()

# with open('/content/ex/classname.txt','r') as inf:
#     class_dict = eval(inf.read()) # class dict 파일

RESULT_PATH = Makedir(RESULT_PATH,'Workspace') # workspace

if args.attention:
    model_name = args.attention.upper()+'_'+args.model.upper()
    RESULT_PATH = Makedir(RESULT_PATH, model_name)

else:
    model_name = args.model.upper()
    RESULT_PATH = Makedir(RESULT_PATH, model_name)

print('Model_name = ',model_name)
print('Result_path = ', RESULT_PATH)

#model set a machine in operation
if args.attention == 'se':
    use_se = True; use_cbam = False
elif args.attention == 'cbam':
    use_se = False; use_cbam = True
else:
    use_se = use_cbam = False    

#model select
models = model_list.Modellist(args.model, class_num, use_se, use_cbam)

#model build
models.build(input_shape=(None, IMAGE_SIZE[0], IMAGE_SIZE[1], IMAGE_SIZE[2]))
print(models.summary())

train_data_gen, valid_data_gen, test_data_gen = make_Data_Loader(IMAGE_PATH = IMAGE_PATH, IMAGE_SHAPE=IMAGE_SIZE, train_batch=train_batch, valid_batch=valid_batch, test_batch=test_batch)


history = model_fit(models, train_data_gen, train_step, args.epochs, valid_data_gen, valid_step, class_num, RESULT_PATH, model_name)
model_test(models, test_data_gen, test_step)
re_dic = model_pred(models, test_data_gen, test_step)


his_graph(history=history, epoch=args.epochs, path=RESULT_PATH)
pred_confusion_matrix(pred=re_dic['pred'], labels=re_dic['labels'], class_num=class_num, path=RESULT_PATH)
# get_feature(model=models, image_path=FEATURE_IMAGE_PATH, result_path=RESULT_PATH, target_size=IMAGE_SIZE)



image_pathes = []
for i in range(40,50):
  path = os.path.join('/content/drive/My Drive/cvpr_data/images/Train_%d.jpg'%i)
  image_pathes.append(path)
  
num_seq = select_seq(args.select_seq)
grad_cam_test2(model=models, seq_num=num_seq, image_paths=image_pathes , save_path=RESULT_PATH, class_names=None, IMAGE_SHAPE=IMAGE_SIZE)

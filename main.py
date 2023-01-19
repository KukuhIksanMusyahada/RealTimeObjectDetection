# Import Stuff
import numpy as np
import cv2
import uuid
import time
import os
import urllib.request
import tarfile
import shutil
import random
import re
import object_detection

from git.repo.base import Repo

# Set things up

np.random.seed(13618045)
min_confidence = 0.3
work_space_path = os.path.dirname( os.path.abspath(__file__) )
parent_path = os.path.join(work_space_path,os.pardir)
PROTOC_PATH = 'C://Program Files//Google Protobuff//'

# images_path = os.path.join(work_space_path ,'images')
# captured_images = os.path.join(images_path,'captured_images')
# colors = np.random.uniform(0,255,size= (len(labels)),3)

CUSTOM_MODEL_NAME = 'efficient_det_models' 
PRETRAINED_MODEL_NAME = 'efficientdet_d0_coco17_tpu-32.tar.gz'
PRETRAINED_MODEL_URL = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d0_coco17_tpu-32.tar.gz'
TF_RECORD_SCRIPT_NAME = 'generate_tfrecord.py'
LABEL_MAP_NAME = 'label_map.pbtxt'

paths = {
    'WORK_SPACE_PATH' : os.path.dirname( os.path.abspath(__file__) ),
    'PARENT_PATH' : os.path.join(os.getcwd(),os.pardir),
    'TF_PATH'   :   os.path.join(parent_path,'TensorFlow'),
    'SCRIPTS_PATH': os.path.join(work_space_path,'TFscripts'),
    'APIMODEL_PATH': os.path.join(parent_path,'TensorFlow','models'),
    'ANNOTATION_PATH': os.path.join(work_space_path,'annotations'),
    'IMAGE_PATH': os.path.join(work_space_path,'images'),
    'CAPTURED_IMAGES_PATH' : os.path.join(work_space_path,'images','captured_imgs'),
    'LABEL_APP_PATH' : os.path.join(work_space_path,'images','label_app'),
    'MODEL_PATH': os.path.join(work_space_path,'models'),
    'PRETRAINED_MODEL_PATH': os.path.join(work_space_path,'models','pre-trained-models'),
    'CHECKPOINT_PATH': os.path.join(work_space_path,'models',CUSTOM_MODEL_NAME), 
    'OUTPUT_PATH': os.path.join(work_space_path,'models',CUSTOM_MODEL_NAME, 'export'), 
    'TFJS_PATH':os.path.join(work_space_path,'models',CUSTOM_MODEL_NAME, 'tfjsexport'), 
    'TFLITE_PATH':os.path.join(work_space_path,'models',CUSTOM_MODEL_NAME, 'tfliteexport'), 
    'PROTOC_PATH':'C://Program Files//Google Protobuff'
 }


files = {
    'PIPELINE_CONFIG':os.path.join(work_space_path,'models', CUSTOM_MODEL_NAME, 'pipeline.config'),
    'TF_RECORD_SCRIPT': os.path.join(os.getcwd(),'SCRIPT', TF_RECORD_SCRIPT_NAME), 
    'LABELMAP': os.path.join(paths['ANNOTATION_PATH'], LABEL_MAP_NAME)
}



# Download and process the data
class CreateTrainingImages:
    def __init__(self,path=paths):
        self.path = path
        self.init_labels = ['Kukuh','Botol']
        self.num_imgs = 5
        self.add_label = True
        self.train_ratio = 0.8

    def addLabel(self):
        while self.add_label:
            add_label = input("add more labels?\n \t1. Yes\n \t2. No \n")
            if add_label == '1':
                new_label = input("Type new label: \n")
                self.init_labels.append(new_label)
            else:
                self.add_label = False
    
    def createDir(self):
        for path in self.path.values():
            if not os.path.exists(path):
                os.mkdir(path)
                
        for label in self.init_labels:
            lab_path = os.path.join(self.path['CAPTURED_IMAGES_PATH'], label)
            if not os.path.exists(lab_path):
                os.mkdir(lab_path)

    def captureImgs(self):
        for label in self.init_labels:
            cap = cv2.VideoCapture(0)
            print(f'Collecting images for {label}')
            time.sleep(5)
            for num_img in range(self.num_imgs):
                print(f"Images collected= {num_img}")
                _,img = cap.read()
                img_name = os.path.join(self.path['CAPTURED_IMAGES_PATH'],label,f'{label}.{str(uuid.uuid1())}.jpg')
                cv2.imwrite(img_name, img)
                cv2.imshow('image',img)
                time.sleep(3)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            cap.release()
            cv2.destroyAllWindows()
    def labelling_imgs(self):
        label_path = self.path['LABEL_APP_PATH']
        print(label_path)
        if not os.path.exists(label_path):
            Repo.clone_from('https://github.com/tzutalin/labelImg', label_path)
        os.chdir(os.path.join(os.getcwd(),'images','label_app'))
        os.system('python labelImg.py')
        os.chdir(self.path['WORK_SPACE_PATH'])
    
    def train_test_splits(self):
        train_path = os.path.join(self.path['IMAGE_PATH'],'train')
        test_path = os.path.join(self.path['IMAGE_PATH'],'test')
        if os.path.exists(train_path):
            shutil.rmtree(train_path)
            os.mkdir(train_path)
        else:
            os.mkdir(train_path)
        if os.path.exists(test_path):
            shutil.rmtree(test_path)
            os.mkdir(test_path)
        else:
            os.mkdir(test_path)
        file_list = list()
        for dir in os.listdir(self.path['CAPTURED_IMAGES_PATH']):
            file_path = os.path.join(self.path['CAPTURED_IMAGES_PATH'],dir)
            for file in os.listdir(file_path):
                if file.endswith('.jpg'):
                    file_name = file[:-4]
                    file_list.append(file_name) 
        file_list = random.sample(file_list,k= int(len(file_list)*self.train_ratio) )
        pattern = '\w*'
        for dir in os.listdir(self.path['CAPTURED_IMAGES_PATH']):
            file_path = os.path.join(self.path['CAPTURED_IMAGES_PATH'],dir)
            for file in os.listdir(file_path):
                if file.endswith('.jpg'):
                    file_name = file[:-4]
                    match_string = re.match(pattern,file_name)
                    if file_name in file_list:
                        shutil.copy(os.path.join(self.path['CAPTURED_IMAGES_PATH'],str(match_string[0]),file_name+'.jpg'),train_path)
                        shutil.copy(os.path.join(self.path['CAPTURED_IMAGES_PATH'],str(match_string[0]),file_name+'.xml'),train_path)
                    else:
                        shutil.copy(os.path.join(self.path['CAPTURED_IMAGES_PATH'],str(match_string[0]),file_name+'.jpg'),test_path)
                        shutil.copy(os.path.join(self.path['CAPTURED_IMAGES_PATH'],str(match_string[0]),file_name+'.xml'),test_path)

            
        
        





    def createData(self):
        self.addLabel()
        self.createDir()
        self.captureImgs()
        self.labelling_imgs()
        self.train_test_splits()
        
        
    










# Get Model

class GetAndTrainModels:
    def __init__(self, paths = paths, files= files):
        self.paths = paths
        self.files = files
        self.training_data = CreateTrainingImages()
        self.labels = self.training_data.init_labels

    def download_all_needed_data(self):
        # download pretrained model
        if not os.path.exists(self.paths['MODEL_PATH']):
            os.mkdir(self.paths['MODEL_PATH'])
        if not os.path.exists(self.paths['PRETRAINED_MODEL_PATH']):
            os.mkdir(self.paths['PRETRAINED_MODEL_PATH'])
        if not os.path.exists(os.path.join(self.paths['PRETRAINED_MODEL_PATH'],PRETRAINED_MODEL_NAME[:-7])):
            urllib.request.urlretrieve(PRETRAINED_MODEL_URL,PRETRAINED_MODEL_NAME)
            tar_ref = tarfile.open(PRETRAINED_MODEL_NAME,'r')
            tar_ref.extractall(self.paths['PRETRAINED_MODEL_PATH'])
            tar_ref.close()
       
    def create_record(self):
        if not os.path.exists(os.path.join(work_space_path,'annotations')):
            os.mkdir(os.path.join(work_space_path,'annotations'))
        with open(self.files['LABELMAP'], 'w') as f:
            for id,label in enumerate(self.labels):
                f.write('item { \n')
                f.write(f'\tname:\'{label}\'\n')
                f.write(f'\tid:{id+1}\n')
                f.write('}\n')
        

    
    def run_models(self):
        pass



#  if not os.path.exists(os.path.join(self.paths['APIMODEL_PATH'], 'research', 'object_detection')):
#             Repo.clone_from("https://github.com/tensorflow/models",self.paths['APIMODEL_PATH'])
#         #download protocol buffer
#         PROTOC_URL = "https://github.com/protocolbuffers/protobuf/releases/download/v21.12/protoc-21.12-win64.zip"
#         # if not os.path.exists(os.path.join(self.paths['PROTOC_PATH'],'protoc-21.12-win64.zip')):
#         #     urllib.request.urlretrieve(PROTOC_URL, 'protoc-21.12-win64.zip')
#             # shutil.move('protoc-21.12-win64.zip',self.paths['PROTOC_PATH'])
#         #extract protocol buffer 
#         # zip_file = os.path.join(self.paths['PROTOC_PATH'],'protoc-21.12-win64.zip')
#         # zip_ref = zipfile.ZipFile(zip_file, 'r')
#         # zip_ref.extractall(self.paths['PROTOC_PATH'])
#         # Set Up Protocol buffer
#         os.environ['PATH'] += os.path.abspath(os.path.join(self.paths['PROTOC_PATH'], 'bin'))
#         os.chdir(os.path.join(self.paths['APIMODEL_PATH'],'research'))
#         os.system('protoc object_detection/protos/*.proto --python_out=.')
#         os.system('copy object_detection\\packages\\tf2\\setup.py setup.py')
#         os.system('python -m pip install --use-feature=2020-resolver')










# Run
if __name__=="__main__":
    # data = CreateTrainingImages()
    # data.createData()
    model = GetAndTrainModels()
    model.download_all_needed_data()
    model.create_record()
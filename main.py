# Import Stuff
import numpy as np
import cv2
import uuid
import time
import os
import traceback
import shutil
import random
import re

from git.repo.base import Repo

# Set things up

np.random.seed(13618045)
min_confidence = 0.3
work_space_path = os.path.dirname( os.path.abspath(__file__) )
# images_path = os.path.join(work_space_path ,'images')
# captured_images = os.path.join(images_path,'captured_images')
# colors = np.random.uniform(0,255,size= (len(labels)),3)

CUSTOM_MODEL_NAME = 'efficient_det_models' 
PRETRAINED_MODEL_NAME = 'efficientdet_d0_coco17_tpu-32'
PRETRAINED_MODEL_URL = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d0_coco17_tpu-32.tar.gz'
TF_RECORD_SCRIPT_NAME = 'generate_tfrecord.py'
LABEL_MAP_NAME = 'label_map.pbtxt'

paths = {
    'WORK_SPACE_PATH' : os.path.dirname( os.path.abspath(__file__) ),
    'SCRIPTS_PATH': os.path.join(work_space_path,'TFscripts'),
    'APIMODEL_PATH': os.path.join(work_space_path,'models'),
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
    'PROTOC_PATH':os.path.join(work_space_path,'protoc')
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
        self.init_labels = []
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

    def create_path(self):
        pass
        

    
    def run_models(self):
        pass













# Run
if __name__=="__main__":
    data = CreateTrainingImages()
    data.createData()
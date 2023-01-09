# Import Stuff
import numpy as np
import cv2
import uuid
import time
import os

from git.repo.base import Repo

# Set things up

np.random.seed(13618045)
min_confidence = 0.3
images_path = os.path.join(os.path.dirname( os.path.abspath(__file__) ),'images')
captured_images = os.path.join(images_path,'captured_images')
# colors = np.random.uniform(0,255,size= (len(labels)),3)



# Download and process the data
class CreateTrainingImages:
    def __init__(self,path=images_path):
        self.path = path
        self.init_labels = []
        self.num_imgs = 20
        self.add_label = True

    def addLabel(self):
        while self.add_label:
            add_label = input("add more labels?\n \t1. Yes\n \t2. No \n")
            if add_label == '1':
                new_label = input("Type new label: \n")
                self.init_labels.append(new_label)
            else:
                self.add_label = False
    
    def createDir(self):
        captured_images_path = os.path.join(self.path,'captured_imgs')
        if not os.path.exists(captured_images_path):
            os.mkdir(captured_images_path)
        for label in self.init_labels:
            lab_path = os.path.join(captured_images_path, label)
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
                img_name = os.path.join(self.path,label,f'{label}.{str(uuid.uuid1())}.jpg')
                cv2.imwrite(img_name, img)
                cv2.imshow('image',img)
                time.sleep(2)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            cap.release()
            cv2.destroyAllWindows()
    def labelling_imgs(self):
        label_path = os.path.join(self.path,'label_imgs')
        if not os.path.exists(label_path):
            Repo.clone_from('https://github.com/tzutalin/labelImg', label_path)
        os.chdir(label_path)
        os.system('pyrcc5 -o libs/resources.py resources.qrc')
        os.system('python labelImg.py')




    def createData(self):
        self.addLabel()
        self.createDir()
        # self.captureImgs()
        self.labelling_imgs()
        
        
    










# Get Model













# Run
if __name__=="__main__":
    data = CreateTrainingImages()
    data.createData()
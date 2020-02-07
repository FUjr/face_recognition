

import sys
import time
import os 
import cv2 
import dlib
import numpy as np
import _thread
import glob

notification_0 = 0

count_0 = 0
get_feature_thread = 0

lasttime = 0

load_0 = 0
load_1 = 0
load_2 = 0

mypath = (__file__[0:-len(os.path.basename(__file__))])
try:
    os.mkdir(mypath+"database")
except:
    a = 1



def load_detector():
    global detector
    global predictor
    detector = dlib.get_frontal_face_detector() 
    predictor_path =mypath+ 'shape_predictor_68_face_landmarks.dat'
    predictor = dlib.shape_predictor(predictor_path)
    global load_0
    load_0 =1
    

def load_model():
    global facerec
    face_rec_model_path = mypath+'dlib_face_recognition_resnet_model_v1.dat'
    facerec = dlib.face_recognition_model_v1(face_rec_model_path)
    global load_1
    load_1 = 1

_thread.start_new_thread (load_detector,())
print("loadding")
cv2.waitKey(1)
_thread.start_new_thread (load_model,())



def get_feature(temp_img,dets,output_name):
    global count_0
    shape = predictor(temp_img, dets[0])
    face_vector = facerec.compute_face_descriptor(temp_img, shape)
    if output_name == 1:
        try:
            os.mkdir(mypath+"database/"+str(people_name))
        except:
                print("had already register")
    file_name = str(time.localtime()[0]) +'.'+str(time.localtime()[1]) +'.' +str(time.localtime()[2]) +'    '+str(time.localtime()[3]) +':'+ str(time.localtime()[4])+'.'+ str(time.localtime()[5]) +'.'+ str(output_name)+".data"
    os.chdir(mypath+"database/"+str(people_name))
    file = open(file_name, mode="w",)
    file.write(str(face_vector))
    print ("feature" + "    "+ str(output_name) + "  "  +"got")
    count_0 = count_0 + 1
    



cv2.waitKey(1)
cap = cv2.VideoCapture(0)
cap.set(3,320) # set Width
cap.set(4,240) # set Height
while not load_0 + load_1 == 2:
    cv2.waitKey(5)
people_name = input("input your name: ")
win = dlib.image_window()



while 1:
        ret , img = cap.read()
        dets = detector(img)
        win.clear_overlay()
        win.set_image(img)
        win.add_overlay(dets)
        if len(dets) > 0 and not get_feature_thread > 3 and time.time() - lasttime > 1 :
            lasttime = time.time()
            get_feature_thread = get_feature_thread + 1
            _thread.start_new_thread(get_feature,(img,dets,get_feature_thread))
        if count_0 >3 :
            break
            
            
dlib.hit_enter_to_continue()

print("finished")
cap.release()     

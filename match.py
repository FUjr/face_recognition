import sys
import time
import os 
import cv2 
import dlib
import numpy as np
import _thread
import glob


d_1 = {}
d_2 = {}
d_3 = {}
d_4 = {}


notification_0 = 0
dict_count = 0
load_0 = 0
load_1 = 0
dect_count = 0
ret = 0
face_ = 0


mypath = (__file__[0:-len(os.path.basename(__file__))])
xml_path = (mypath + 'haarcascade_frontalface_default.xml')
def detector(img):
    face0 = []
    face1 = cv2.CascadeClassifier(xml_path)
    face2 = face1.detectMultiScale(
        cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),

        scaleFactor=1.2,
        minNeighbors = 10
        ,
    )
    for a,b,c,d in face2:
        face0.append(dlib.rectangle(a,b,a+c,b+d))
    return face0
def load_detector():
    #global detector
    global predictor
    #detector = dlib.get_frontal_face_detector() 
    predictor_path =mypath +  'shape_predictor_68_face_landmarks.dat'
    predictor = dlib.shape_predictor(predictor_path)
    global load_0
    load_0 =1
    

def load_model():
    global facerec
    face_rec_model_path = mypath + 'dlib_face_recognition_resnet_model_v1.dat'
    facerec = dlib.face_recognition_model_v1(face_rec_model_path)
    global load_1
    load_1 = 1


def get_feature(img,dets):
    shape = predictor(img, dets)
    face_vector = facerec.compute_face_descriptor(img, shape)
    return face_vector


def list_file():
    os.chdir(mypath + "database")
    file_list = os.listdir(mypath + "database")
    return file_list
def list_data(data):
    file_list = os.listdir(mypath + "database/"+ str(data))
    return file_list

def task(file_list,dictionary):
    for task in file_list:
        match(task,dictionary)


def diff(a):
    global face
    a,face = np.array(a), np.array(face)
    sub = np.sum((a-face)**2)
    add = (np.sum(a**2)+np.sum(face**2))/2.
    return sub/add


    
def match(task,dictionary):
    match = []
    average_score = 0
    count_file = 0
    data = list_data(task)
    for file_name in data:
        file = open(mypath + "database/"+task+'/' +str(file_name),'r')
        temp_list = file.readlines()
        count_file = count_file + 1
        file.close()
        for i in temp_list:
            match.append(float(i))
        diffe = diff(match)
        match.clear()
        average_score = average_score + diffe
        if dictionary == "d_1":
            d_1[task] = average_score/count_file
        elif dictionary == "d_2":
            d_2[task] = average_score/count_file
        elif  dictionary == "d_3":
            d_3[task] = average_score/count_file
        elif  dictionary == "d_4":
            d_4[task] = average_score/count_file

            
def get_min(dict_name):
    list_d_v = []
    list_d_k = []
    if dict_name == 'd_1':
        for i in d_1.keys():
            list_d_k.append(i)
        for i in d_1.values():
            list_d_v.append(i)
            number = list_d_v.index(min(list_d_v))
            min_ = min(list_d_v)
    elif dict_name == 'd_2':
        for i in d_2.keys():
            list_d_k.append(i)
        for i in d_2.values():
            list_d_v.append(i)
            number = list_d_v.index(min(list_d_v))
            min_ = min(list_d_v)
    elif dict_name == 'd_3':
        for i in d_3.keys():
            list_d_k.append(i)
        for i in d_3.values():
            list_d_v.append(i)
            number = list_d_v.index(min(list_d_v))
            min_ = min(list_d_v)
    elif dict_name == 'd_4':
        for i in d_4.keys():
            list_d_k.append(i)
        for i in d_4.values():
            list_d_v.append(i)
            number = list_d_v.index(min(list_d_v))
            min_ = min(list_d_v)
    if min_ > 0.1:
        return min_ , 'unkown'
    return min_ , list_d_k[number]
    
def final():
    if dict_count == 0:
        min_ , final = get_min('d_1')
    if dict_count == 4:
        min_1 , final_1 = get_min('d_1')
        min_2 , final_2 = get_min('d_2')
        min_3 , final_3 = get_min('d_3')
        min_4 , final_4 = get_min('d_4')
        list_v = [min_1,min_2,min_3,min_4]
        number = list_v.index(min(list_v))
        list_k = [final_1,final_2,final_3,final_4]
        final = list_k[number]
    return final , min(list_v)

print("loadding")
_thread.start_new_thread (load_detector,())
_thread.start_new_thread (load_model,())



while not load_0 + load_1 == 2:
    cv2.waitKey(10)


print("loaded")
cv2.waitKey(1)
cap = cv2.VideoCapture(0)
#cap.set(3,300) # set Width
#cap.set(4,200) # set Height
def dete():
    global face
    deted = []
    for i, d in enumerate(dets):
                e = [d.left(),d.top()]
                face = get_feature(img,dets[i])
                task(list_file(),"d_1")
                n , name = get_min('d_1')
                deted.append(name)
                
    return deted


dets_ = 0
dets_1 = 0
dets_2 = 0
count_round = 0
count_time = time.time()
while 1:
    dets_2 =dets_2 + 1
    ret , show_img = cap.read()
    img = cv2.resize(show_img,(300,200))
    cv2.waitKey(10)
    time1 = time.time()

    if  time.time()- dets_1 > 0.5:
        dets_1 =time.time()
        time1 = time.time()
        dets = detector(img)
#        print(str(time.time() - time1) + '  people_det')
    if len(dets) > 0:
        if len(dets) == dets_:
            time1 = time.time()
            for i,d in enumerate(dets):
               e = [int(d.left()*2.13),int(d.top()*2.13)]
#               cv2.circle(show_img,tuple([int(((d.right()- d.left())/2*2.13)+d.left()),int((((d.top() -d.bottom())/2)+d.bottom())*2.13)]), int((d.right() - d.left())/2*2.13), (0,0,255), -1)
               cv2.rectangle(show_img, tuple([int(d.left()*2.13),int( d.top()*2.13)]), tuple([int(d.right()*2.13), int(d.bottom()*2.13)]), (0, 255, 255), 2)
               cv2.putText(show_img, name[i], tuple(e), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
#            print(str(time.time() - time1) + '  same_people_det')
        else:
            time1 = time.time()
            name = dete()
            for i,d in enumerate(dets):
                e = [int(d.left()*2.13),int(d.top()*2.13)]
#                cv2.circle(show_img,tuple([int(((d.right()- d.left())/2*2.13)+d.left()),int((((d.top() -d.bottom())/2)+d.bottom())*2.13)]), int((d.right() - d.left())/2*2.13), (0,0,255), -1)
                cv2.rectangle(show_img, tuple([int(d.left()*2.13),int( d.top()*2.13)]), tuple([int(d.right()*2.13), int(d.bottom()*2.13)]), (0, 255, 255), 2)
                cv2.putText(show_img, name[i], tuple(e), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
#                print(str(time.time() - time1) + '  new_people_det')
                dets_ = len(dets)

    else:
        dets_ = 0
    
#    print(time.time() - round_time )
    cv2.imshow('detector',show_img)
    count_round = count_round + 1
    if time.time() - count_time  > 3:
        count_time =time.time()
        print(str(count_round/3)+'fps')
        count_round = 0
        


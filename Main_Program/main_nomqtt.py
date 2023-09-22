### IMPORT LIBRARY ###
import cv2 as cv
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import pickle
from keras_facenet import FaceNet

import gspread
from oauth2client.service_account import ServiceAccountCredentials
from pprint import pprint

import datetime

import time

import pyvisa

### IMPORT LIBRARY ###

### INITIALIZE  MODEL ###
facenet = FaceNet()
faces_embeddings = np.load("faces_embeddings_test_EC(300).npz")
Y = faces_embeddings['arr_1']
encoder = LabelEncoder()
encoder.fit(Y)
haarcascade = cv.CascadeClassifier("haarcascade_frontalface_default.xml")
model = pickle.load(open("svm_model_EC_160x160(300).pkl", 'rb'))
### INITIALIZE  MODEL ###

### INITIALIZE  FOR GOOGLE SHEET ###
scope = ["https://spreadsheets.google.com/feeds",'https://www.googleapis.com/auth/spreadsheets',
         "https://www.googleapis.com/auth/drive.file","https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_name("creds.json", scope)

client = gspread.authorize(creds)
#sheet = client.open("Python_Test").worksheet("ชีต1")
sheet = client.open("Python_Test").worksheet('ชีต1')
### INITIALIZE  FOR GOOGLE SHEET ###

### INITIALIZE  FOR SERIAL DATA ###
ports = pyvisa.ResourceManager()
print(ports.list_resources())

serialPort = ports.open_resource('ASRL12::INSTR')
serialPort.baud_rate=9600
### INITIALIZE  FOR SERIAL DATA ###

### VARIABLE ###
cap = cv.VideoCapture(1)
list_name = []
row_X = 2
colum_Y = 1

state = 0

messageIn = ""
buffer_name = ""
#st_send = 0
### VARIABLE ###

# WHILE LOOP
while cap.isOpened():
    if(serialPort.bytes_in_buffer):
        while(serialPort.bytes_in_buffer):
            time.sleep(0.1)
            messageIn = serialPort.read_bytes(serialPort.bytes_in_buffer)
        if(messageIn == b'ON'):
            print("BUFFER SERIAL:ON")
            state = 1
        elif(messageIn == b'OFF'):
            print("BUFFER SERIAL:OFF")
            values_list = list(sheet.get_all_values())
            sheet.update_cell(len(values_list) + 1,colum_Y,"-------")
            state = 0
            
    if state == 1:
        _, frame = cap.read()
        rgb_img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        gray_img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        faces = haarcascade.detectMultiScale(gray_img, 1.3, 5)

        for x,y,w,h in faces:
            img = rgb_img[y:y+h, x:x+w]
            img = cv.resize(img, (80,80)) # 1x160x160x3
            img = np.expand_dims(img,axis=0)
            ypred = facenet.embeddings(img)
            face_name = model.predict(ypred)
            proba = ypred[0]
            final_name = encoder.inverse_transform(face_name)[0]
            percent = np.round((1+proba)/2 *100, 2)
            #pprint(final_name + " " + str(percent[0]))
            if percent[0] >= 53.5:
                pprint(final_name + " " + str(percent[0]))
                cv.putText(frame, str(final_name) + " " + str(percent), (x,y-10), cv.FONT_HERSHEY_SIMPLEX,
                        1, (0,0,255), 3, cv.LINE_AA)
                        
                if final_name not in list_name:
                    buffer_name = final_name.encode()

                    serialPort.write_raw(b"t1.txt=\"")
                    serialPort.write_raw(buffer_name)
                    serialPort.write_raw(b"\"")
                    serialPort.write_raw(b"\xff\xff\xff")
                    
                    list_name.append(final_name)
                    values_list = list(sheet.get_all_values())
                    print(len(values_list)+1)
                    sheet.update_cell(len(values_list) + 1,colum_Y,final_name)
                    sheet.update_cell(len(values_list) + 1,colum_Y+1,str(datetime.datetime.now()))

        cv.imshow("Face Recognition:", frame)
    
    elif state == 0:
        del list_name[:]
        
        buffer_name = ("STUDENT:").encode()
        serialPort.write_raw(b"t1.txt=\"")
        serialPort.write_raw(buffer_name)
        serialPort.write_raw(b"\"")
        serialPort.write_raw(b"\xff\xff\xff")
        
        _, frame = cap.read()
        cv.imshow("Face Recognition:", frame)

    if cv.waitKey(1) == ord('q'):
        break

cv.waitKey(0)
cap.release()
cv.destroyAllWindows()

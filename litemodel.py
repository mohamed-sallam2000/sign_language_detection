import cv2
import serial
import numpy as np
import threading
import tensorflow as tf
from googletrans import Translator
from pygame import mixer
import mediapipe as mp
import time
import os 
import threading
from qtpy.QtWidgets import QApplication, QLabel, QWidget, QPushButton
from qtpy.QtGui import QImage, QPixmap
from qtpy.QtCore import QTimer
import pyttsx3
actions = np.array(["hello", "thanks","open"])

mp_holistic = mp.solutions.holistic
sequence = []
sentence = []
predictions = []
threshold = 0.4#نسبه  
interval =30
thread_flag=False
offlag=False
onflag=False

#(some code)   global flag =0   (other code)
cap_1 = cv2.VideoCapture(0)

frame_width = int(cap_1.get(3))
frame_height = int(cap_1.get(4))

size=(frame_width,frame_height)
print(size)
interpreter = tf.lite.Interpreter(model_path="twoexp.tflite")
interpreter.allocate_tensors()
app = QApplication([])
window = QWidget()
window.setStyleSheet("background-color: black;")
alarm_window=QWidget()
alarm_window.resize(1366,768)
alarm_window.setStyleSheet("background-color: red;") 
label = QLabel(window)
label.setGeometry(0, 0, frame_width, frame_height)#find size
label_2 = QLabel(window)
label_2.setGeometry(0, 400, 800,150)#600-480->timer=120
label_2.setStyleSheet("background-color: black; color: #ccc;")

label_3 = QLabel(window)
label_3.setGeometry(0, 600, 800,150)#600-480->timer=120
label_3.setStyleSheet("background-color: black; color: #ccc;")
timer = QTimer(timeout=lambda:update_timer_label())
timer.setInterval(1000)  # Set the timer to run every 1 second
time_var =0
timer_label = QLabel(window)
timer_label.setGeometry(0, 480, 100, 100)
timer_label.setStyleSheet("background-color: black; color: #ccc;")
label_alarm = QLabel(alarm_window)
label_alarm.setGeometry(0, 0, 490, 600)
mic_label = QLabel(window)
mic_label.setGeometry(900, 50, 300, 100)
mic_label.setStyleSheet("background-color: black; color: #ccc;")
start_button = QPushButton("Start ", window,clicked=lambda:press_fun())
start_button.setGeometry(800, 490, 300, 80)#300 & 490 & 300 & 80
start_button.setStyleSheet("background-color: black ; color: #ccc;")
reconnect_btn = QPushButton("reconnect ", window,clicked=lambda:try_connect())
reconnect_btn.setGeometry(800, 650, 300, 80)
reconnect_btn.setStyleSheet("background-color: black ; color: #ccc;")
reconnect_btn.hide()
try:
    ser=serial.Serial('COM6', 9600, timeout=1)
except:
    ser=None
    mic_label.setText("not connect ")
    reconnect_btn.show()
# light_button=QPushButton("on",window,clicked=lambda:setonflag())
# light_button.setGeometry(800, 490, 300, 80)
# light_button.setStyleSheet("background-color: black ; color: #ccc;")
# off_button=QPushButton("off",window,clicked=lambda:setofflag())
# off_button.setGeometry(800, 650, 300, 80)
# off_button.setStyleSheet("background-color: black ; color: #ccc;")

trans_button = QPushButton("translate ", window,clicked=lambda:translate(sentence))
trans_button.setGeometry(800, 600, 300, 80)
trans_button.setStyleSheet("background-color: black ; color: #ccc;")
def update_timer_label():
    global time_var
    time_var += 1
    if time_var <= interval and thread_flag:
        timer_label.setText("{} seconds".format(time_var))
    else:
        pass
        # call function
        # timer_label.setText("Time's up!")
        # timer.stop()
def press_fun():

    global thread_flag
         

 
    
    
    global time_var  
    time_var = 0
   
    thread_flag=True  
    label_3.setText("start recording ...")
    
    # record()

        # t2.stop()
def translate(txt):
    translator = Translator(service_urls=['translate.googleapis.com'])
    mixer.init()
    # Translate the text
    converter = pyttsx3.init()
    txt=''.join(txt)
    translated_text = translator.translate(txt, dest='en').text
    #print(translated_text)

    # convert into speech
    #speech = gTTS(text=translated_text, lang='en', slow=False)
    #speech.save("txt.mp3")
    converter.setProperty('rate', 150)
    # Set volume 0-1
    converter.setProperty('volume', 0.7)
    converter.say(translated_text)

    
    # Empties the say() queue
    # Program will not continue
    # until all speech is done talking
    converter.runAndWait()
    # mixer.music.load('txt.mp3')
    # mixer.music.play() #Playing Music with Pygame
    # while mixer.music.get_busy():
    #     continue
    # mixer.music.load('hello.mp3')
    # os.remove('txt.mp3')
    # mixer.music.stop()
    label_3.setText( translated_text)
def try_connect():
    global ser
    try:
        ser=serial.Serial('COM6', 9600, timeout=1)
    except:
        pass    
def model_fun():
    global actions,sequence ,sentence ,predictions
   
    def mediapipe_detection(image, model):
        image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
        image.flags.writeable = False 
                        # Image is no longer writeable
        results = model.process(image)                 # Make prediction
        image.flags.writeable = True                   # Image is now writeable 
    #  image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
        return image, results
    def extract_keypoints(results):
        
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
        face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
        lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
        rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
        return np.concatenate([pose, face, lh, rh])  
   
    
   
    input_details=interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    with mp_holistic.Holistic() as holistic:
        
        cap = cv2.VideoCapture('001.avi')
     
        while(cap.isOpened()):
            #app.processEvents()
            
            ret, frame = cap.read()
        
            if ret:
                
                image, results = mediapipe_detection(frame, holistic)
                keypoints = extract_keypoints(results)
                sequence.append(keypoints)#math array numpy 
                sequence = sequence[-30:]#اخر 30عنصر  
                
                if len(sequence) == 30:
                    #res = model.predict(np.expand_dims(sequence, axis=0))[0]
                    input=np.array(np.expand_dims(sequence, axis=0),dtype=np.float32)
                    interpreter.set_tensor(input_details[0]['index'],input )
                   
                    interpreter.invoke()
                   
                    res = interpreter.get_tensor(output_details[0]['index'])[0]
                   
                    
                    predictions.append(np.argmax(res))
                    # image_1 = QImage(frame, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
                    # label.setPixmap(QPixmap.fromImage(image_1))
                    if np.unique(predictions[-10:])[0]==np.argmax(res): 
                        if res[np.argmax(res)] > threshold:     
                            if len(sentence) > 0: 
                                if actions[np.argmax(res)] != sentence[-1]:
                                    # label_2.setText(actions[np.argmax(res)])
                                    
                                    sentence.append(actions[np.argmax(res)])    
                            else:
                            
                                sentence.append(actions[np.argmax(res)])
                    if len(sentence) > 10: 
                        sentence = sentence[-10:]   
            else:
                break                   
        # image_1 = QImage(frame, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
        # label.setPixmap(QPixmap.fromImage(image_1))
     
        label_2.setText(''.join(sentence))  
        label_3.setText("record is done  ... click translate ") 
        # print(sentence)
        cap.release()
        

     
                     
def serial_thread():
    global onflag,offlag
    def on_fun():
        ser.write(b'o')  
        time.sleep(0.1)
    

    def off_fun():
        ser.write(b'f')  
        time.sleep(0.1)
        
    # try:
    #         #windows put="COM6"

    #         ser=serial.Serial('COM6', 9600, timeout=1)
            
    # except:
    #         mic_label.setText("cant connect to serial devise  ")
    #         if(reconnect_btn.isHidden()):       
    #             reconnect_btn.show()

    while 1:
     
        try:
            #windows put="COM6"
            # ser.reset_input_buffer()
            # ser.reset_output_buffer()
            # time.sleep(0.01)
           # ser=serial.Serial('COM6', 9600, timeout=1)
            # ser.read_all()
            # ser.flushOutput()
            # print(ser.readline())
            data =str(ser.readline())[2:-5]
            
            # data=str(ser.readline())
            if 'i'in data:
                label_alarm.setText("alarm")
                
                alarm_window.show()
            if onflag:

                #print("onfla")
                on_fun()
                onflag=False  
              
            if offlag:
                off_fun()  
                offlag=False  

            
            # print("mic")
            mic_label.setText('noise intensty ='+str(data)) 
            reconnect_btn.hide()   
            # time.sleep(0.005) 

        except:
            #ser.close()
            
            mic_label.setText("cant connect to serial devise  ") 
            if(reconnect_btn.isHidden()):       
                reconnect_btn.show()
                # time.sleep(0.5)    
def record():
    cap = cv2.VideoCapture(0)    
            
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    size = (frame_width, frame_height)
    result = cv2.VideoWriter('001.avi',cv2.VideoWriter_fourcc(*'XVID'),10, size)
                
    global interval 
    global time_var   
  

    global thread_flag
    while 1:

        if thread_flag:
            cap = cv2.VideoCapture(0)    
          
            while(time_var<interval):
                ret, frame = cap.read()
                #print("insde loop")
                if ret == True: 
            
            
                    result.write(frame)
            
                    
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image_1 = QImage(frame, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
                    label.setPixmap(QPixmap.fromImage(image_1))
            thread_flag=False
            cap.release()    
            result.release()
            model_fun()
            

                
    # Closes all the frames
#main threaD+SERIAL THREAD + recording threAd
# 
#    
t2=threading.Thread(target=record)
t1 = threading.Thread(target=serial_thread)
t1.start()  
# with mp_holistic.Holistic() as holistic:
#     while cap.isOpened():
t2.start()
timer.start() 
while 1:
   
    
     
    app.processEvents()
    window.show()  
    
 ###########################3
 # thread one always on 
 #  gui 
 # prograss bar
 # ai model for  phrase generating from discrete words  decrease responce 
 # save as frames not avi numpyfiles 
 # timer.show and strart recording  
 # solve lag
 # maximum words 3 due to structure data set     
 #value each 200ms(verical progress bar )
 #model -->20
# 10 models 
# translate --> list[]
# install package in pi 4
# mediapipe bluesye 
# install new reaspine (legacy)
# bash script 
# python 3.7







    
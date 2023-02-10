
import cv2
#import serial
import numpy as np
import tensorflow as tf
#from matplotlib import pyplot as plt
import mediapipe as mp
#from tensorflow import keras
#from tensorflow.keras.utils.np_utils import to_categorical
#from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.models import load_model
    
from qtpy.QtWidgets import QApplication, QLabel, QWidget, QPushButton
from qtpy.QtGui import QImage, QPixmap
from qtpy.QtCore import QTimer
actions = np.array(['مرحبا ', 'thanks'])
mp_holistic = mp.solutions.holistic
model = load_model('twoexp.h5',compile=False)
sequence = []
sentence = []
predictions = []
threshold = 0.5#نسبه  
interval =30
cap = cv2.VideoCapture(0)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
size = (frame_width, frame_height)
interpreter = tf.lite.Interpreter(model_path="twoexp.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
#ser=serial.Serial('COM6', 9600, timeout=1)
#mic==label 3
#door =window red background
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

################################################################################
################################################################################
app = QApplication([])
window = QWidget()
window.width()
window.height()
#window.size()getter
window.setStyleSheet("background-color: white;")
alarm_window=QWidget()
alarm_window.resize(1366,768)
alarm_window.setStyleSheet("background-color: red;")####مستني الي يناديه  
# Create a QLabel to display the video
label = QLabel(window)
label.setGeometry(0, 0, frame_width, frame_height)
label_2 = QLabel(window)
label_2.setGeometry(0, 600, 400,150)#600-480->timer=120
label_2.setStyleSheet("background-color: white; color: black;")
# Start the video capture
# Create a QTimer
timer = QTimer()
timer.setInterval(1000)  # Set the timer to run every 1 second
time_var =0
# Create aQLabel to display the timer
timer_label = QLabel(window)
timer_label.setGeometry(0, 480, 100, 100)
timer_label.setStyleSheet("background-color: white; color: black;")

label_alarm = QLabel(alarm_window)
label_alarm.setGeometry(0, 0, frame_width, frame_height)

mic_label = QLabel(window)
mic_label.setGeometry(600, 480, 100, 100)
mic_label.setStyleSheet("background-color: white; color: black;")
mic_label.show()
def update_timer_label():
    global time_var
    time_var += 1
    if time_var <= interval:
        timer_label.setText("{} seconds".format(time_var))
    else:
        # call function
        timer_label.setText("Time's up!")
        time_var = 0
        timer.stop()
timer.timeout.connect(update_timer_label)
def press_fun():
    
    start_button.setText("translate now")
 
  
  
#def on_fun():
   # ser.write(b'o')  

   

#def off_fun():
    #ser.write(b'f')  
    


#x=0        
# Create a start button
start_button = QPushButton("Start ", window,clicked=lambda:press_fun())
start_button.setGeometry(300, 490, 300, 80)
start_button.setStyleSheet("background-color: white ; color: black;")
light_button=QPushButton("on",window)
light_button.setGeometry(800, 490, 300, 80)
light_button.setStyleSheet("background-color: white ; color: black;")
off_button=QPushButton("off",window)
off_button.setGeometry(800, 650, 300, 80)
off_button.setStyleSheet("background-color: white ; color: black;")
start_button.clicked.connect(timer.start)
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ''' data =str(ser.readline())[2:-5]
        print(data)
        if 'i'in data:
            x=1
    # Capture frame-by-frame
        if x==1:
            label_alarm.setText("alarm")
            app.processEvents()
            alarm_window.show()
            '''

       
        
     
        ret, frame = cap.read()
       

        image, results = mediapipe_detection(frame, holistic)
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)#math array numpy 
        sequence = sequence[-30:]#اخر 30عنصر  
        #print(sequence)
        if len(sequence) == 30:
            #res = model.predict(np.expand_dims(sequence, axis=0))[0]
            input=np.array(np.expand_dims(sequence, axis=0),dtype=np.float32)
            interpreter.set_tensor(input_details[0]['index'],input )
            interpreter.invoke()
            res = interpreter.get_tensor(output_details[0]['index'])[0]
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            predictions.append(np.argmax(res))
            image_1 = QImage(frame, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
            label.setPixmap(QPixmap.fromImage(image_1))
            if np.unique(predictions[-10:])[0]==np.argmax(res): 
                if res[np.argmax(res)] > threshold:     
                    if len(sentence) > 0: 
                        if actions[np.argmax(res)] != sentence[-1]:
                            label_2.setText(actions[np.argmax(res)])
                            
                            sentence.append(actions[np.argmax(res)])    
                    else:
                       
                        sentence.append(actions[np.argmax(res)])
            if len(sentence) > 5: 
                sentence = sentence[-5:]   
        image_1 = QImage(frame, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
        label.setPixmap(QPixmap.fromImage(image_1))
        app.processEvents()
        window.show()    
        mic_label.setText("noise level= ")    
        







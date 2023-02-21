import cv2
import time

font = cv2.FONT_HERSHEY_SIMPLEX
  
# org
org = (50, 50)
  
# fontScale
fontScale = 1
   
# Blue color in BGR
color = (255, 0, 0)
  
# Line thickness of 2 px
thickness = 2
word=0
cap=cv2.VideoCapture(0)
interval=3#time before capture 
start_counting_img=0#to know how many imges you made 
words=["hello","thanks","hear","thirsty"]
img_num=start_counting_img
curent_time=int(time.time())

while 1:
    remain=int(time.time())-curent_time
    ret,frame=cap.read()
    if ret:
      
        if remain>=interval:
            curent_time=int(time.time())
            cv2.imwrite('{}{}.jpg'.format(words[word],img_num), frame)
            img_num+=1

            if img_num>=50:
                img_num=0
                word+=1
        image = cv2.putText(frame, 'word:{}{} will captuer after :{}'.format(words[word],img_num,(interval-remain)), org, font, 
                fontScale, color, thickness, cv2.LINE_AA)
        cv2.imshow("data set maker",image)        
            

        




    if cv2.waitKey(25) & 0xFF == 0x20:#  ord(''): when user press space program 
     input("Press the <Enter> key to continue...")
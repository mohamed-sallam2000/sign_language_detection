from qtpy.QtWidgets import QApplication, QLabel, QWidget, QPushButton
from qtpy.QtGui import QImage, QPixmap
from qtpy.QtCore import QTimer

import cv2
cap=cv2.VideoCapture(0)

app=QApplication([])#empty list equal sys.argv
#used in command lines any argument you pass to 
# the script will save in this  empty list
#like wrinting in cmd {$myscript.py --info}
#the list will contain --info  
#there are some commands are 
#called by this way isnot in our consern 
#see https://www.pythonguis.com/faq/qapplication-sysargv-command-line-arguments/
window = QWidget()#make object of  class qwidget
window.setGeometry(0,0,400,400)#مفروض اي عنصر اعمله استخدم فانكشن  د
#window.setGeometry(x_orgine,y_orgine,width ,higth)
#if set geometery deos not called then a defult values will be obtained
#or use another method called resize 
window.resize(500,500)#why not size?--->size is getter not setter(oop)
#resize--> setter 
#size  --->getter
window.setStyleSheet("background-color: black;")
start_button = QPushButton("start timer ", window,clicked=lambda:press())#clicked=press()
start_button.setGeometry(300, 100, 300, 80)
start_button.setStyleSheet("  background-color: black;color:#ccc;;border-style: outset;border-width: 2px ;border-radius: 20px;font: bold 14px;")
timer_label = QLabel(window)
timer_label.setGeometry(0, 0, 100, 100)
timer_label.setStyleSheet("background-color: black; color: #ccc;font: bold 14px;")
timer_label.setText("time")
#مكان يتحط فيه  نص لايمكن تعديله بواسطه يوزر
timer = QTimer(timeout=lambda:update_timer_label())#make new object of class timer
timer.setInterval(1000)#كل ثانيه هينفذ امر معين 
interval=30#كل 30ثانيه هينفذ امر تاني 
period=10#كل 10 ثواني  هينفذ امر تالت 
time_var=0
#فانكشن  بتتنفذ كل ثانيه
def update_timer_label():
    global time_var#=0
    timer_label.setText("{} seconds".format(time_var))
    time_var += 1#زود واحد كل مرة يتم تنفيذ الداله يعني كل ثانيه 
    if time_var <= period:#طالما اقل من 10 اعرض الوقت
        timer_label.setText("{} seconds".format(time_var))
        timer_label.setGeometry(timer_label.x(),timer_label.y()+5,timer_label.width(),timer_label.height())
    elif time_var <= interval and time_var>period:#طالما اقل من  30 واكبر من عشرة  
        timer_label.setGeometry(timer_label.x()+5,timer_label.y(),timer_label.width(),timer_label.height())#
        #يحرك النص كل ثانيه خمسه بكسل 
        timer_label.setText("{} seconds".format(time_var))#حرك النص كل ثانيه    
    else:
        # call function
        timer_label.setText("Time's up!")
        time_var = 0
        timer_label.setGeometry(0,0,timer_label.width(),timer_label.height())#رجع النص نقطه بدايه
        timer.stop()
  

def press():
    timer.start()   
#timer.timeout.connect(update_timer_label)#نفذ  الداله عند انتهاء العداد ==كل ثانيه 
label=QLabel(window)#عشان اعرض فديو 
label.setGeometry(640,500,500,500)

while 1:
    #cv2 return image with BGR so we use formate BGR888
    ret,frame=cap.read()#بقرأ صورة من الكاميرا واخزن ف  فريم 
    print(frame.shape) #will equal (480, 640, 3)(hight,width ,colors)
#ابعاد الفديو ()
    image_1 = QImage(frame, frame.shape[1], frame.shape[0], QImage.Format_BGR888)
   # image_1=QImage(frame,width,hight,image formate)
   #كدا معايا نص فارغ اسمه (label)
   #ومعايا صورة image_1
   #عايز ادمجهم
    label.setPixmap(QPixmap.fromImage(image_1))
    #label.setText("text")
    window.show() #will show also all cheldern classes
    app.processEvents()#make gui responce  work in separate thread 
    
#x = lambda a : a + 10#بتصنع فانكشن#
#print(x(5))


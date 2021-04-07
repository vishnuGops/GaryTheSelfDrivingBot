import RPi.GPIO as GPIO    
from gpiozero import Motor
import time

#set the GPIO pins of raspberry pi.
GPIO.setmode (GPIO.BCM)
GPIO.setwarnings (False)
#enable
GPIO.setup(16, GPIO.OUT)
GPIO.setup(20, GPIO.OUT)
#setting the GPIO pin as Output
GPIO.setup (24, GPIO.OUT)
GPIO.setup (23, GPIO.OUT)
GPIO.setup (27, GPIO.OUT)
GPIO.setup (22, GPIO.OUT)
#GPIO.PWM( pin, frequency ) it generates software PWM
PWMR = GPIO.PWM (24, 100)
PWMR1 = GPIO.PWM (23, 100)
PWML = GPIO.PWM (27, 100)
PWML1 = GPIO.PWM (22, 100)
#Starts PWM at 0% dutycycle
PWMR.start (0)
PWMR1.start (0)
PWML.start (0)
PWML1.start (0)
#enable pins of the motor
GPIO.output(16, GPIO.HIGH)
GPIO.output(20, GPIO.HIGH)

motor1 = Motor(24, 23)
motor2 = Motor(27, 22)


#GPIO Mode (BOARD / BCM)
GPIO.setmode(GPIO.BCM)
 
#set GPIO Pins
GPIO_TRIGGER = 18
GPIO_ECHO = 25
 
#set GPIO direction (IN / OUT)
GPIO.setup(GPIO_TRIGGER, GPIO.OUT)
GPIO.setup(GPIO_ECHO, GPIO.IN)





import cv2
import numpy as np

def distance():
    # set Trigger to HIGH
    GPIO.output(GPIO_TRIGGER, True)
 
    # set Trigger after 0.01ms to LOW
    time.sleep(0.00001)
    GPIO.output(GPIO_TRIGGER, False)
 
    StartTime = time.time()
    StopTime = time.time()
 
    # save StartTime
    while GPIO.input(GPIO_ECHO) == 0:
        StartTime = time.time()
 
    # save time of arrival
    while GPIO.input(GPIO_ECHO) == 1:
        StopTime = time.time()
 
    # time difference between start and arrival
    TimeElapsed = StopTime - StartTime
    # multiply with the sonic speed (34300 cm/s)
    # and divide by 2, because there and back
    distance = (TimeElapsed * 34300) / 2
 
    return distance


capture = cv2.VideoCapture(0)  #read the video
capture.set(3,320.0) #set the size
capture.set(4,240.0)  #set the size
capture.set(5,15)  #set the frame rate

for i in range(0,2):
        flag, trash = capture.read() #starting unwanted null value

while cv2.waitKey(1) != 27:
        dist = distance()
        frame=capture.g
        
        flag, frame = capture.read() #read the video in frames
        
        lower = [0, 0, 0]
        upper = [70, 70, 70]
        
        lower_red = [17, 15, 100]
        upper_red = [50, 56, 200]

        # create NumPy arrays from the boundaries
        lower = np.array(lower, dtype="uint8")
        upper = np.array(upper, dtype="uint8")
        lower_red = np.array(lower_red, dtype="uint8")
        upper_red = np.array(upper_red, dtype="uint8")

        # find the colors within the specified boundaries and apply
        # the mask
        mask = cv2.inRange(frame, lower, upper)
        mask_red = cv2.inRange(frame, lower_red, upper_red)
        output = cv2.bitwise_and(frame, frame, mask=mask)

        ret, th1 = cv2.threshold(mask, 40, 255, 0)
        ret, th2 = cv2.threshold(mask_red, 40, 255, 0)
        _, contours, hierarchy = cv2.findContours(th1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        _, contours_red, hierarchy = cv2.findContours(th2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for cnt_red in contours_red:
           if cnt_red is None:
                break
           if cnt_red is not None:
            area = cv2.contourArea(cnt_red)# find the area of contour
           if area>=300 :
            cv2.drawContours(frame, cnt_red, -1, (255, 0, 0), 3)
            cv2.imshow('frame', frame)
            M = cv2.moments(cnt_red)
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            if cx>130 and cx<190:
                print("Destination Reached- Put U Turn")
                motor2.forward(0.9)
                motor1.backward(0.9)
                time.sleep(1.2)
         
        for cnt in contours:
           #if cnt is None:
                   #break
           if cnt is not None:
            area = cv2.contourArea(cnt)# find the area of contour
           if area>=1000 :
            cv2.drawContours(frame, cnt, -1, (0, 255, 0), 3)
            cv2.imshow('frame', frame)  # show video
            # find moment and centroid
            M = cv2.moments(cnt)
            cx = int(M['m10']/M['m00'])

            cy = int(M['m01']/M['m00'])
      
            if dist <= 15:
                print ("Stop")
                print ("Measured Distance = %.1f cm" % dist)
                motor1.forward(0)
                motor2.forward(0)
                time.sleep(0.08)
                
            elif cx<=145:
                print("left")
                #l=(cx*100/160)
                motor1.forward(0.3)
                motor2.backward(0.25)
                time.sleep(.03)
                
            elif cx>=175 and cx<190:
                print("right")
                #r=((320-cx)*100/160)
                motor2.forward(0.3)
                motor1.backward(0.25)
                time.sleep(.03)
                
            elif cx>=190 :
                print("right Turn")
                motor1.forward(0)
                motor2.forward(0)
                time.sleep(0.08)
                motor2.forward(0.5)
                motor1.forward(0.5)
                time.sleep(3.75)
                motor2.forward(0.5)
                motor1.backward(0.5)
                time.sleep(3.1)
                
                
               
            elif cx>145 and cx<175:
                print("straight")
                motor1.forward(0.5)
                motor2.forward(0.5)
                time.sleep(0.1)
                
            else:
                motor1.forward(0.5)
                motor2.forward(0.5)
                time.sleep(.08)
                
           
            
        motor1.stop()
        motor2.stop()
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
         break

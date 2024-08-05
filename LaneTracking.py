# LANE DETECTION:

import math
import cv2
import numpy as np
import time
from picamera2 import Picamera2
import RPi.GPIO as GPIO          
from time import sleep

in1 = 16
in2 = 18
in3 = 11
in4 = 13
ena = 22
enb = 15

temp1 = 1

GPIO.setmode(GPIO.BOARD) 
GPIO.setup(in1, GPIO.OUT)
GPIO.setup(in2, GPIO.OUT)
GPIO.setup(in3, GPIO.OUT)
GPIO.setup(in4, GPIO.OUT)
GPIO.setup(ena, GPIO.OUT)
GPIO.setup(enb, GPIO.OUT)

p1 = GPIO.PWM(ena, 5000)
p2 = GPIO.PWM(enb, 5000)

GPIO.output(in1, GPIO.LOW)
GPIO.output(in2, GPIO.LOW)
GPIO.output(in3, GPIO.LOW)
GPIO.output(in4, GPIO.LOW)

p1.start(25)
p2.start(25)
p1.ChangeDutyCycle(75)
p2.ChangeDutyCycle(75)

# Define the motor movement functions
def move_forward():

    GPIO.output(in1, GPIO.LOW)
    GPIO.output(in2, GPIO.HIGH)
    GPIO.output(in3, GPIO.LOW)
    GPIO.output(in4, GPIO.HIGH)

def move_backward():
    GPIO.output(in1, GPIO.HIGH)
    GPIO.output(in2, GPIO.LOW)
    GPIO.output(in3, GPIO.HIGH)
    GPIO.output(in4, GPIO.LOW)

def move_left():
    GPIO.output(in1, GPIO.LOW)
    GPIO.output(in2, GPIO.HIGH)
    GPIO.output(in3, GPIO.HIGH)
    GPIO.output(in4, GPIO.LOW)

def move_right():
    GPIO.output(in1, GPIO.HIGH)
    GPIO.output(in2, GPIO.LOW)
    GPIO.output(in3, GPIO.LOW)
    GPIO.output(in4, GPIO.HIGH)
    
def slope(vx1, vx2, vy1, vy2):
    """
    Calculate the slope angle between two points.
    """
    m = float(vy2 - vy1) / float(vx2 - vx1)
    theta1 = math.atan(m)
    return theta1 * (180 / np.pi)
    
def process_image(image):
    a = b = c = 1  # Initialize variables a, b, and c
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    equ = cv2.equalizeHist(gray)
    blur = cv2.GaussianBlur(equ, (5, 5), 0)
    ret, thresh = cv2.threshold(blur, 240, 255, cv2.THRESH_BINARY)

    # Find Contours
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Draw Contour
    cv2.drawContours(thresh, contours, -1, (255, 0, 0), 3)

    drawing = np.zeros(image.shape, np.uint8)

    lines = cv2.HoughLinesP(thresh, 1, np.pi/180, 25, minLineLength=10, maxLineGap=40)

    l = r = 0
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                if round(x2 - x1) != 0:
                    arctan = slope(x1, x2, y1, y2)
                    if 250 < y1 < 600 and 250 < y2 < 600:
                        if round(-80) <= round(arctan) <= round(-30):
                            r += 1
                            l = 0
                            cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2, cv2.LINE_AA)
                        elif round(30) <= round(arctan) <= round(80):
                            l += 1
                            r = 0
                            cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2, cv2.LINE_AA)

    if l >= 10 and a == 1:
        print('left')
        move_left()
        a = 0
        b = 1
        c = 1
    elif r >= 10 and b == 1:
        print('right')
        move_right()
        a = 1
        b = 0
        c = 1
    elif l < 10 and r < 10 and c == 1:
        print('straight')
        move_forward()
        a = 1
        b = 1
        c = 0

    # Display the images
    cv2.imshow("Threshold", thresh)
    cv2.imshow("Frame", image)

    key = cv2.waitKey(1) & 0xFF

if __name__ == '__main__':
    camera = Picamera2()
    camera.start()
    try:
        while True:
            image = camera.capture_array()  # Directly capture the image
            process_image(image)
    finally:
        camera.stop()
        cv2.destroyAllWindows()

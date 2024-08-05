# REMOTE MOTOR CONTROLLER


import RPi.GPIO as GPIO
from time import sleep

# Motor Control GPIO Pins
in1 = 16
in2 = 18
in3 = 11
in4 = 13
ena = 22
enb = 15

# Initialize PWM
p1 = GPIO.PWM(ena, 3000)
p2 = GPIO.PWM(enb, 3000)

# Set GPIO mode and setup motor control pins
GPIO.setmode(GPIO.BOARD)
GPIO.setup([in1, in2, in3, in4, ena, enb], GPIO.OUT)

# Start PWM
p1.start(75)
p2.start(75)

print("\n")
print("The default speed & direction of motor is LOW & Forward.....")
print("r - run, s - stop, f - forward, b - backward, l - low, m - medium, h - high, ri - right, le - left, e - exit")
print("\n")

while True:
    x = input()

    if x == 'r':
        print("run")
        if temp1 == 1:
            GPIO.output(in1, GPIO.LOW)
            GPIO.output(in2, GPIO.HIGH)
            GPIO.output(in3, GPIO.LOW)
            GPIO.output(in4, GPIO.HIGH)
            print("forward")
        else:
            GPIO.output(in1, GPIO.HIGH)
            GPIO.output(in2, GPIO.LOW)
            GPIO.output(in3, GPIO.HIGH)
            GPIO.output(in4, GPIO.LOW)
            print("backward")

    elif x == 's':
        print("stop")
        GPIO.output(in1, GPIO.LOW)
        GPIO.output(in2, GPIO.LOW)
        GPIO.output(in3, GPIO.LOW)
        GPIO.output(in4, GPIO.LOW)

    elif x == 'f':
        print("forward")
        GPIO.output(in1, GPIO.LOW)
        GPIO.output(in2, GPIO.HIGH)
        GPIO.output(in3, GPIO.LOW)
        GPIO.output(in4, GPIO.HIGH)
        temp1 = 1

    elif x == 'b':
        print("backward")
        GPIO.output(in1, GPIO.HIGH)
        GPIO.output(in2, GPIO.LOW)
        GPIO.output(in3, GPIO.HIGH)
        GPIO.output(in4, GPIO.LOW)
        temp1 = 0

    elif x == 'l':
        print("low")
        p1.ChangeDutyCycle(25)
        p2.ChangeDutyCycle(25)

    elif x == 'm':
        print("medium")
        p1.ChangeDutyCycle(50)
        p2.ChangeDutyCycle(50)

    elif x == 'h':
        print("high")
        p1.ChangeDutyCycle(75)
        p2.ChangeDutyCycle(75)
        
    elif x== 'ri':
        print("right")
        GPIO.output(in1, GPIO.LOW)
        GPIO.output(in2, GPIO.LOW)
        GPIO.output(in3, GPIO.LOW)
        GPIO.output(in4, GPIO.HIGH)
        
    elif x=='le':
        print("left")
        GPIO.output(in1, GPIO.LOW)
        GPIO.output(in2, GPIO.HIGH)
        GPIO.output(in3, GPIO.LOW)
        GPIO.output(in4, GPIO.LOW)        

    elif x == 'e':
        GPIO.cleanup()
        print("GPIO Clean up")
        break

    else:
        print("<<<  wrong data  >>>")
        print("please enter the defined data to continue.....")

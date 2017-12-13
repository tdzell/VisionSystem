import numpy as np
import cv2
import sharing
import RPI.GPIO as GPIO

def SoundAlarm():

GPIO.setmode(GPIO.BOARD)
GPIO.setup(14, GPIO.OUT, initial=GPIO.LOW)
GPIO.output(14, 1)	
cv2.waitKey(250)
GPIO.output(14, 0)
GPIO.cleanup()

def GlobeCreate(): ###initializes all global variables that need to be tracked in this module

    
    boltCount = 0
    
    global oneBoltSeen
    oneBoltSeen = 0
    global twoBoltSeen
    twoBoltSeen = 0
    global threeBoltSeen
    threeBoltSeen = 0
    global fourBoltSeen
    fourBoltSeen = 0
    

    outerCount = 0
    global oneOuterSeen
    oneOuterSeen = 0
    global twoOuterSeen
    twoOuterSeen = 0

    handleCount = 0
    global handleSeen
    handleSeen = 0
    
    global noBoltSeen
    noBoltSeen = 0
    
    global counterimage
    counterimage = 0

def AlarmDetect(DetectedClasses, ClassNames, imgToBeSaved):
    
    ###declare that the following variables are global variables, and to not treat their names as new local variables within this function
    global counterimage
    boltCount = 0
    global oneBoltSeen #Seen variables used to mark how many times within the last X frames a given detection has occured
    global twoBoltSeen
    global threeBoltSeen
    global fourBoltSeen
    outerCount = 0
    global oneOuterSeen
    global twoOuterSeen
    handleCount = 0
    global handleSeen
    global noBoltSeen
    global blurredimg
    
    Counter = [[],[]] #initialization of a list so that it can be appended
    
    
    for Name in ClassNames: # initialize structure for the list
        
        Counter[0].append(Name)
        Counter[1].append(0)

    for DetectedName in DetectedClasses: #count how many times each classid appears in the given array of detected classes
        
        Counter[1][DetectedName] = Counter[1][DetectedName] + 1
    
    ### pull the number of detections for classes we care about into human readable variables
    boltCount = Counter[1][0]
    outerCount = Counter[1][2]
    handleCount = Counter[1][1] 
    
        if boltCount > 0 :
        SoundAlarm()
        
    return 'nothing', False
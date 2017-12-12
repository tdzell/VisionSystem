import numpy as np
import cv2
import sharing

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
    
    saveimage = False
    
    for Name in ClassNames: # initialize structure for the list
        
        Counter[0].append(Name)
        Counter[1].append(0)

    for DetectedName in DetectedClasses: #count how many times each classid appears in the given array of detected classes
        
        Counter[1][DetectedName] = Counter[1][DetectedName] + 1
    
    ### pull the number of detections for classes we care about into human readable variables
    boltCount = Counter[1][0]
    outerCount = Counter[1][2]
    handleCount = Counter[1][1] 
    
    if boltCount == 0 :
        noBoltSeen += 1 # if no bolts have been seen, increment this counter | if a judgement needs to occur, it will only happen once no bolts have been seen for X frames 
    if boltCount == 1:
        oneBoltSeen += 1 #increment this number of bolts seen
        noBoltSeen = 0 #reset the reset counter | if a judgement needs to occur, it will only happen once no bolts have been seen for X frames
    if boltCount == 2:
        twoBoltSeen += 1
        noBoltSeen = 0
    if boltCount == 3:
        threeBoltSeen += 1
        noBoltSeen = 0
    if boltCount == 4:
        fourBoltSeen += 1
        noBoltSeen = 0
    if boltCount > 4: #for the current application, no more than 4 bolts should ever possibly be seen at one time
        print('ERROR: MORE THAN 4 BOLTS SEEN')
    
    if outerCount == 1 : 
        oneOuterSeen += 1 
        noBoltSeen = 0 #reset the reset counter | if a judgement needs to occur, it will only happen once no bolts have been seen for X frames
    elif outerCount == 2:
        twoOuterSeen += 1
        noBoltSeen = 0
    if outerCount > 2:
        print('ERROR: MORE THAN 2 OUTER SEEN')

    if handleCount == 1:
        handleSeen += 1
        noBoltSeen = 0
        
    if handleCount > 1:
        print('ERROR: MORE THAN 1 HANDLE SEEN')
    print('no bolt: %s' % (noBoltSeen))
    colorframe = 'nothing'
    
    if (oneBoltSeen + twoBoltSeen + threeBoltSeen + fourBoltSeen) == 1:
        
        sharing.holdimg = imgToBeSaved
    
    
    if noBoltSeen > 8: #if no bolts have been seen for X frames, check if alarm needs to be raised, otherwise return to the calling script
        boltExpected = 0
    
        if twoOuterSeen >= sharing.detect_min:
            boltExpected += 2 #if two "outer signs" have been detected, there should be two associated bolts that were also seen
        elif oneOuterSeen >= sharing.detect_min:
            boltExpected += 1 #if one "outer sign" has been detected, there should be one associated bolt that was also seen
        
        if handleSeen >= sharing.detect_min:
            boltExpected += 2 #if a "handle" has been detected, than there should be two associated bolts that were also seen
    
        if fourBoltSeen >= sharing.detect_min: #if four bolts were seen at once for at least 'sharing.detect_min' times, then we assume that four bolts were seen
            boltSeen = 4
        elif threeBoltSeen >= sharing.detect_min: #etc for three bolts
            boltSeen = 3
        elif twoBoltSeen >= sharing.detect_min: #etc for two bolts
            boltSeen = 2
        elif oneBoltSeen >= sharing.detect_min: #etc for one bolt
            boltSeen = 1
        else:
            boltSeen = 0 #else no bolts were seen
        
        if boltExpected != 0:
            if boltSeen > boltExpected:
                print('||CONFUSED:, %s bolts seen but %s expected' % (boltSeen, boltExpected))
                sharing.saveimage = True #mark that the held image should be saved to file by the calling script
                sharing.colorframe = 'yellow' #mark that a yellow screen should be returned instead of normal detection feed
            if boltSeen == boltExpected:
                print('VERIFIED, %s bolts expected' % (boltExpected))
                sharing.colorframe = 'green' #mark that a green screen should be returned instead of normal detection feed
            if boltSeen < boltExpected:
                print('**ALARM**, %s bolts seen but %s expected' % (boltSeen, boltExpected))
                sharing.saveimage = True #mark that the held image should be saved to file by the calling script
                sharing.colorframe = 'red' #mark that a red screen should be returned instead of normal detection feed
            #cv2.waitKey(0)
        else:
            sharing.colorframe = 'nothing'
            
        noBoltSeen = 0
        oneBoltSeen = 0
        twoBoltSeen = 0
        threeBoltSeen = 0
        fourBoltSeen = 0
        oneOuterSeen = 0
        twoOuterSeen = 0
        handleSeen = 0
        
    return colorframe, saveimage
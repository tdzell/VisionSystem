import numpy as np
import cv2
import sharing

def GlobeCreate():


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

def AlarmDetect(DetectedClasses, ClassNames):
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
	
	Counter = [[],[]]
	
	saveimage = False
	
	for Name in ClassNames:
		
		Counter[0].append(Name)
		Counter[1].append(0)

	for DetectedName in DetectedClasses: #count how many times each classid appears in the given array of detected classes
		
		Counter[1][DetectedName] = Counter[1][DetectedName] + 1
		
	boltCount = Counter[1][0] 
	outerCount = Counter[1][2]
	handleCount = Counter[1][1] #check to ensure correctness
	
	if boltCount == 0 :
		noBoltSeen += 1
	if boltCount == 1:
		oneBoltSeen += 1 #increment this number of bolts seen
		noBoltSeen = 0 #reset the reset counter
	if boltCount == 2:
		twoBoltSeen += 1
		noBoltSeen = 0
	if boltCount == 3:
		threeBoltSeen += 1
		noBoltSeen = 0
	if boltCount == 4:
		fourBoltSeen += 1
		noBoltSeen = 0
	if boltCount > 4:
		print('ERROR: MORE THAN 4 BOLTS SEEN')
	
	if outerCount == 1 :
		oneOuterSeen += 1
		noBoltSeen = 0
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
	if noBoltSeen > 8: #if no bolts have been seen for X frames, check if alarm needs to be raised, reset otherwise
		boltExpected = 0
	
		if twoOuterSeen >= sharing.detect_min:
			boltExpected += 2
		elif oneOuterSeen >= sharing.detect_min:
			boltExpected += 1
		
		if handleSeen >= sharing.detect_min:
			boltExpected += 2
	
		if fourBoltSeen >= sharing.detect_min:
			boltSeen = 4
		elif threeBoltSeen >= sharing.detect_min:
			boltSeen = 3
		elif twoBoltSeen >= sharing.detect_min:
			boltSeen = 2
		elif oneBoltSeen >= sharing.detect_min:
			boltSeen = 1
		else:
			boltSeen = 0
		
		if boltExpected != 0:
			if boltSeen > boltExpected:
				print('CONFUSED: MORE BOLTS SEEN THAN EXPECTED')
				print(boltSeen)
				print(boltExpected)
				#cv2.waitKey(0)
				saveimage = True
				colorframe = 'yellow'
				print('expected: %s' % (boltExpected))
			if boltSeen == boltExpected:
				print('VERIFIED')
				#cv2.waitKey(0)
				colorframe = 'green'
				print('expected: %s' % (boltExpected))
			if boltSeen < boltExpected:
				print('**ALARM**, %s bolts seen but %s expected' % (boltSeen, boltExpected))
				print(boltSeen)
				print(boltExpected)
				saveimage = True
				colorframe = 'red'
				#cv2.waitKey(0)
		else:
			colorframe = 'nothing'
			
		noBoltSeen = 0
		oneBoltSeen = 0
		twoBoltSeen = 0
		threeBoltSeen = 0
		fourBoltSeen = 0
		oneOuterSeen = 0
		twoOuterSeen = 0
		handleSeen = 0
		
	return colorframe, saveimage
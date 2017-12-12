from utils import *
from darknet import Darknet
import cv2
import AlarmDetector


import numpy as np
import multiprocessing
from multiprocessing import Queue, Pool
from threading import Thread
from ctypes import byref
import sharing

def IDSCamera(cfgfile, weightfile, useGPU):
    from pyueye_example_camera import Camera
    #from pyueye_example_utils import *
    from pyueye import ueye #importing this too early would require IDS camera drivers to be installed just to run the "StandardCamera" code
    
    ### initialization for creation of a .avi file for sharing of proof of concept
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter('output.avi',fourcc,4.0,(640,480))
	
    
    ### IDS camera initializations
    cam = Camera()
    cam.init()
    cam.set_colormode(ueye.IS_CM_BGR8_PACKED)
    cam.alloc()
    cam.capture_video() 
    
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter('output.avi',fourcc,4.0,(640,480))
    num_workers = 2
    input_q = Queue(2)
    output_q = Queue(2)
    pool = Pool(num_workers, IDS_worker, (input_q, output_q, cfgfile, weightfile, useGPU))
       
    timeout = 1000
    running = True
    m = Darknet(cfgfile)
    m.print_network()
    m.load_weights(weightfile)
    sharing.usegpu = useGPU
    sharing.loop = True
        
        
    if m.num_classes == 20:
        namesfile = 'data/voc.names'
    elif m.num_classes == 80:
        namesfile = 'data/coco.names'
    else:
        namesfile = 'data/names'
        
    m.class_names = load_class_names(namesfile)
 
        
    if useGPU:
        m.cuda()
    print('Loading weights from %s... Done!' % (weightfile))

    while running:
        img_buffer = ImageBuffer()
        ret = ueye.is_WaitForNextImage(cam.handle(),
                                       timeout,
                                       img_buffer.mem_ptr,
                                       img_buffer.mem_id)
                                           
        if ret == ueye.IS_SUCCESS:
            input_q.put(ImageData(cam.handle(), img_buffer))
    
        if cv2.waitKey(1) & 0xFF == ord('q'):
            running = False
            sharing.loop = False
            
            cam.stop_video()
            cam.exit()
            pool.terminate()
            cv2.destroyAllWindows()
            break
    
        img, bboxes = output_q.get()
        print('------')
        draw_img, waitsignal = plot_boxes_cv2(img, bboxes, None, class_names) #draw boxes associated with detections onto the base images | AlarmDetection.py is called in here
        cv2.imshow('cfgfile', draw_img) #show the image frame that now has detections drawn onto it | draw_image will be entirely green/yellow/red after a judgement is made by AlarmDetection.py for verification or alarm
        
        '''uncomment the following line to record video | file is named output.avi and will overwrite any existing files with same name'''        
        #out.write(draw_img)
		
        cv2.waitKey(100)
		
    

def IDSworker(input_q, output_q, cfgfile, weightfile, useGPU):

    sharing.detect_min = 3
    sharing.colorframe = 'nothing'
    sharing.saveimage  = False
    sharing.counterimage = 0
    
    if useGPU:
        sharing.usegpu = True
    else:
        sharing.usegpu = False
    
    ### initialization of neural network based upon the specified config and weights files
    m = Darknet(cfgfile)
    m.print_network()
    m.load_weights(weightfile)
    
        
    
    ### if GPU optimizations are enabled, do some initialization
    if sharing.usegpu:
        m.cuda()
    print('Loading weights from %s... Done!' % (weightfile))

    
    while True:
        
        input_q.get(image)
        image = image_data.as_1d_image()
        image_data.unlock()
        sized = cv2.resize(image, (m.width, m.height))
        bboxes = do_detect(m, sized, 0.4, 0.4, useGPU) #third value in this call sets the confidence threshold for object detection
        print('------')
        output_q.put((img, do_detect(m, sized, 0.4, 0.4, useGPU)))
        #out.write(draw_img)
            

        if waitsignal == True:
            cv2.waitKey(2000)
            waitsignal = False
            
        
               
    
    
    
    
def StandardCamera(cfgfile, weightfile, useGPU):
    m = Darknet(cfgfile)
    ### initialization for creation of a .avi file for sharing of proof of concept
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter('output.avi',fourcc,4.0,(640,480))
    
    ### initialization of pulling image frames from generic USB camera using openCV
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    if not cap.isOpened():
        print("Unable to open camera")
        exit(-1)

    num_workers = 2
    input_q = Queue(2)
    output_q = Queue(2)
    res, img = cap.read()
            
    input_q.put(img)  
    pool = Pool(num_workers, Standard_worker, (input_q, output_q, cfgfile, weightfile, useGPU))
        
    if m.num_classes == 20:
            namesfile = 'data/voc.names'
    elif m.num_classes == 80:
            namesfile = 'data/coco.names'
    else:
            namesfile = 'data/names'

    class_names = load_class_names(namesfile)    
        
    while True:
        
        res, img = cap.read()
            
        input_q.put(img)
        
        img, bboxes = output_q.get()
        print('------')
        draw_img, waitsignal = plot_boxes_cv2(img, bboxes, None, class_names) #draw boxes associated with detections onto the base images | AlarmDetection.py is called in here
        cv2.imshow('cfgfile', draw_img) #show the image frame that now has detections drawn onto it | draw_image will be entirely green/yellow/red after a judgement is made by AlarmDetection.py for verification or alarm
        
        '''uncomment the following line to record video | file is named output.avi and will overwrite any existing files with same name'''        
        #out.write(draw_img)
        
        if waitsignal == True: #if green/yellow/red screen is being shown by draw_img, leave it in place for two seconds instead of continuing detections
            cv2.waitKey(2000)
            waitsignal = False
    
        cv2.waitKey(350) #neccessary to ensure this loop does not attempt to pull new images from the USB camera too quickly
        
            
        
  #      if cv2.waitKey(1) & 0xFF == ord('q'):
  #          break
        
    pool.terminate()
    cap.stop()
    cv2.destroyAllWindows()
        
        
   
def Standard_worker(input_q, output_q, cfgfile, weightfile, useGPU):
    
    
    sharing.detect_min = 3
    sharing.colorframe = 'nothing'
    sharing.saveimage  = False
    sharing.counterimage = 0
    
    if useGPU:
        sharing.usegpu = True
    else:
        sharing.usegpu = False
    
    ### initialization of neural network based upon the specified config and weights files
    m = Darknet(cfgfile)
    m.print_network()
    m.load_weights(weightfile)
    
        
    
    ### if GPU optimizations are enabled, do some initialization
    if sharing.usegpu:
        m.cuda()
    print('Loading weights from %s... Done!' % (weightfile))
    
    while True:

        
        img = input_q.get()
        
        sized = cv2.resize(img, (m.width, m.height)) #resize the image frame pulled into the size expecteed by the detection model
        #bboxes = do_detect(m, sized, 0.4, 0.4, useGPU) #third value in this call sets the confidence needed to detect object
            
        output_q.put((img, do_detect(m, sized, 0.4, 0.4, useGPU)))
            

    
                

if __name__ == '__main__':

    AlarmDetector.GlobeCreate() #initializes module level global counters for AlarmDetector.py

    ### initialization of program level global variables for: configuration; saving of "falsepositive" images
    sharing.detect_min = 3
    sharing.colorframe = 'nothing'
    sharing.saveimage  = False
    sharing.counterimage = 0
    
    ### exactly four arguments must be present after calling this python script in command prompt for the rest of the script to run
    if len(sys.argv) == 5:
        cfgfile = sys.argv[1] #pulling the arguments given from command prompt
        weightfile = sys.argv[2]
        cpuGPU = sys.argv[3]
        cameraUsage = sys.argv[4]
        
        
        if cpuGPU == 'GPU':
            useGPU = True
        else:
            useGPU = False
        
        if useGPU:
            sharing.usegpu = True
        else:
            sharing.usegpu = False
        
        if cameraUsage == 'IDS': #If "IDS" is the final argument given, use the IDS Camera code, otherwise use the generic USB camera code
            IDSCamera(cfgfile, weightfile, useGPU)
        else:
            StandardCamera(cfgfile, weightfile, useGPU)
        #demo('cfg/tiny-yolo-voc.cfg', 'tiny-yolo-voc.weights')
    else:
        print('Usage:')
        print('    python demo.py cfgfile weightfile')
        print('')
        print('    perform detection on camera')

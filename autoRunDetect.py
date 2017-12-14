# Python external libraries
import cv2 #OpenCV
import numpy as np

# Python default libraries
from multiprocessing import Queue, Pool
from threading import Thread
import time
import glob

# Python modules
from utils import plot_boxes_cv2, do_detect, load_class_names
from darknet import Darknet
import sharing
import AlarmDetector


def IDSCamera(cfgfile, weightfile, useGPU):

    from ctypes import byref
    
    # Python external libraries
    from pyueye import ueye #importing this too early would require IDS camera drivers to be installed just to run the "StandardCamera" code
    
    # Python modules
    from pyueye_example_camera import Camera
    from pyueye_example_utils import ImageData, Rect, ImageBuffer
    
    
    
    ### IDS camera initializations
    cam = Camera()
    cam.init()
    cam.set_colormode(ueye.IS_CM_BGR8_PACKED)
    cam.alloc()
    cam.capture_video() 
    
        
    ### startup of thread that pulls image frames from the IDS camera
    input_q = Queue(8)
    output_q = Queue(8)
    thread = FrameThread(cam, 1, cfgfile, weightfile, useGPU, input_q, output_q)
    thread.start()
    loop = True
    
    m = Darknet(cfgfile)
    ### initialization for creation of a .avi file for sharing of proof of concept
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter('output.avi',fourcc,5.0,(480, 360))
    
    if m.num_classes == 20:
            namesfile = 'C:/Users/Catharina/Documents/GitHub/VisionSystem/data/voc.names'
    elif m.num_classes == 80:
            namesfile = 'C:/Users/Catharina/Documents/GitHub/VisionSystem/data/coco.names'
    else:
            namesfile = 'data/names'
     
        
    class_names = load_class_names(namesfile)
     
    num_workers = 2
    pool = Pool(num_workers, IDS_worker, (input_q, output_q, cfgfile, weightfile, useGPU))
    

    
    while loop:
        cv2.waitKey(10)
        image, bboxes = output_q.get()
        
        print('------')
        draw_img, waitsignal = plot_boxes_cv2(image, bboxes, None, class_names) #draw boxes associated with detections onto the base images | AlarmDetection.py is called in here
        cv2.imshow('cfgfile', draw_img) #show the image frame that now has detections drawn onto it | draw_image will be entirely green/yellow/red after a judgement is made by AlarmDetection.py for verification or alarm
        '''uncomment the following line to record video | file is named output.avi and will overwrite any existing files with same name'''        
        #out.write(draw_img)
        
        if waitsignal == True:
            cv2.waitKey(2000)
            waitsignal = False
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            loop = False
            out.release()
            cv2.destroyAllWindows()
            thread.stop()
            thread.join()
            print('join')
            pool.terminate()
            print('terminate')
            cam.stop_video()
            print('stop_video')
            cam.exit()
            print('cam exit')

            break
            
    print('IDS_Camera close')
        
    

def IDS_worker(input_q, output_q, cfgfile, weightfile, useGPU):

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
        
        image = input_q.get()
        sized = cv2.resize(image, (m.width, m.height))
    #third value in this call sets the confidence threshold for object detection
        output_q.put((image, do_detect(m, sized, 0.4, 0.4, useGPU)))
        #out.write(draw_img)
                  
    
    
    
    
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

    num_workers = 1
    input_q = Queue(4)
    output_q = Queue(4)

            
 
    pool = Pool(num_workers, Standard_worker, (input_q, output_q, cfgfile, weightfile, useGPU))
        
    if m.num_classes == 20:
            namesfile = 'C:/Users/Catharina/Documents/GitHub/VisionSystem/data/voc.names'
    elif m.num_classes == 80:
            namesfile = 'C:/Users/Catharina/Documents/GitHub/VisionSystem/data/coco.names'
    else:
            namesfile = 'C:/Users/Catharina/Documents/GitHub/VisionSystem/data/names'

    class_names = load_class_names(namesfile)    
    
    cv2.namedWindow('cfgfile', cv2.WND_PROP_FULLSCREEN)          
    cv2.setWindowProperty('cfgfile', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    for imgname in glob.glob('*.bmp'):
        
        img = cv2.imread(imgname,1)   
        input_q.put(img)
        
        img, bboxes = output_q.get()
        print('------')
        draw_img, waitsignal = plot_boxes_cv2(img, bboxes, None, class_names) #draw boxes associated with detections onto the base images | AlarmDetection.py is called in here
        cv2.imshow('cfgfile', draw_img) #show the image frame that now has detections drawn onto it | draw_image will be entirely green/yellow/red after a judgement is made by AlarmDetection.py for verification or alarm
        
        '''uncomment the following line to record video | file is named output.avi and will overwrite any existing files with same name'''        
        cv2.imwrite("DrawnHold/drawn_" + imgname,img)
        
        if waitsignal == True: #if green/yellow/red screen is being shown by draw_img, leave it in place for two seconds instead of continuing detections
            cv2.waitKey(2000)
            waitsignal = False
    
        cv2.waitKey(3) #neccessary to ensure this loop does not attempt to pull new images from the USB camera too quickly
        time.sleep(0.05)
            
        
  #      if cv2.waitKey(1) & 0xFF == ord('q'):
  #          break
        
    pool.terminate()
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
            



        
    
    
    
    
    

class FrameThread(Thread):
    def __init__(self, cam, views, cfgfile, weightfile, useGPU, input_q, output_q, copy=True):
        super(FrameThread, self).__init__()
        self.timeout = 1000
        self.cam = cam
        self.running = True
        self.views = views   
        sharing.usegpu = useGPU
        sharing.loop = True
        
        self.input_q = input_q
        self.output_q = output_q

        
    def run(self):

        while self.running:
            img_buffer = ImageBuffer()
            ret = ueye.is_WaitForNextImage(self.cam.handle(),
                                           self.timeout,
                                           img_buffer.mem_ptr,
                                           img_buffer.mem_id)
            if ret == ueye.IS_SUCCESS:
                self.notify(ImageData(self.cam.handle(), img_buffer))
            cv2.waitKey(100)
            time.sleep(0.01)

    

    def notify(self, image_data):

        if self.views:
            if type(self.views) is not list:
                self.views = [self.views]
            
            for view in self.views:
 
                image = image_data.as_1d_image()
                image_data.unlock()
                self.input_q.put(image)

    def stop(self):
        
        self.running = False
        
                

if __name__ == '__main__':
    
    AlarmDetector.GlobeCreate() #initializes module level global counters for AlarmDetector.py

    ### initialization of program level global variables for: configuration; saving of "falsepositive" images
    sharing.detect_min = 3
    sharing.colorframe = 'nothing'
    sharing.saveimage  = False
    sharing.counterimage = 0
    
    cfgfile = 'C:/Users/Catharina/Documents/GitHub/VisionSystem/cfg/yolo-voc.cfg'
    weightfile = 'C:/Users/Catharina/Documents/GitHub/VisionSystem/HenrikTest2.weights'
    cpuGPU = 'GPU'
    cameraUsage = 'Standard'
        
            
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



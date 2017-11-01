from utils import *
from darknet import Darknet
import cv2
import AlarmDetector

import pyueye
from pyueye import ueye
from pyueye_example_camera import Camera
from pyueye_example_utils import *
import numpy as np
from threading import Thread
from ctypes import byref

def demo(cfgfile, weightfile):
    #GlobeCreate()
    
    
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter('output.avi',fourcc,30.0,(640,480))
    
    
    
    
    m = Darknet(cfgfile)
    cam = Camera()
    cam.init()
    #ueye.is_SetBinning(cam.h_cam, (ueye.IS_BINNING_3X_VERTICAL or ueye.IS_BINNING_3X_HORIZONTAL))
    cam.set_colormode(ueye.IS_CM_BGR8_PACKED)
    #cam.set_aoi(0,0, 600, 480)
    cam.alloc()
    cam.capture_video()
    
    '''
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    if not cap.isOpened():
        print("Unable to open camera")
        exit(-1) 
    '''
    thread = FrameThread(cam, 1, cfgfile, weightfile)
    thread.start()
    '''
    while True:

        
        res, img = cap.read()
        
        
        if res:
            sized = cv2.resize(img, (m.width, m.height))
            bboxes = do_detect(m, sized, 0.4, 0.4) #third value in this call sets the confidence needed to detect object?
            print('------')
            draw_img = plot_boxes_cv2(img, bboxes, None, class_names)
            cv2.imshow(cfgfile, draw_img)
            #out.write(draw_img)
            cv2.waitKey(1)
        else:
             print("Unable to read image")
             exit(-1)
             
'''
class FrameThread(Thread):
    def __init__(self, cam, views, cfgfile, weightfile, copy=True):
        super(FrameThread, self).__init__()
        self.timeout = 1000
        self.cam = cam
        self.running = True
        self.views = views
        self.copy = copy
        self.m = Darknet(cfgfile)
        self.m.print_network()
        self.m.load_weights(weightfile)
        
        if self.m.num_classes == 20:
            namesfile = 'data/voc.names'
        elif self.m.num_classes == 80:
            namesfile = 'data/coco.names'
        else:
            namesfile = 'data/names'
        
        self.m.class_names = load_class_names(namesfile)
 
        use_cuda = 1
        if use_cuda:
            self.m.cuda()
        print('Loading weights from %s... Done!' % (weightfile))
    def run(self):

        while self.running:
            img_buffer = ImageBuffer()
            ret = ueye.is_WaitForNextImage(self.cam.handle(),
                                           self.timeout,
                                           img_buffer.mem_ptr,
                                           img_buffer.mem_id)
                                           
            if ret == ueye.IS_SUCCESS:
                self.notify(ImageData(self.cam.handle(), img_buffer))


    def notify(self, image_data):

        if self.views:
            if type(self.views) is not list:
                self.views = [self.views]
            
            for view in self.views:
 
                image = image_data.as_1d_image()
                image_data.unlock()
                sized = cv2.resize(image, (self.m.width, self.m.height))
                bboxes = do_detect(self.m, sized, 0.4, 0.4) #third value in this call sets the confidence needed to detect object?
                print('------')
                draw_img = plot_boxes_cv2(image, bboxes, None, self.m.class_names)
                cv2.imshow(cfgfile, draw_img)
                
                cv2.waitKey(200)
              
                
############################################
if __name__ == '__main__':
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
    GlobeCreate()


    if len(sys.argv) == 3:
        cfgfile = sys.argv[1]
        weightfile = sys.argv[2]
        demo(cfgfile, weightfile)
        #demo('cfg/tiny-yolo-voc.cfg', 'tiny-yolo-voc.weights')
    else:
        print('Usage:')
        print('    python demo.py cfgfile weightfile')
        print('')
        print('    perform detection on camera')

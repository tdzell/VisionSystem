from utils import *
from darknet import Darknet
import cv2
import AlarmDetector

def demo(cfgfile, weightfile):
    AlarmDetector.GlobeCreate()
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
    global counterimage
    counterimage = 0
    m = Darknet(cfgfile)
    m.print_network()
    m.load_weights(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))
    
    sharing.detect_min = 3
    sharing.usegpu = False
    createglobal()
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter('output.avi',fourcc,3.0,(640,480))
    
    if m.num_classes == 20:
        namesfile = 'data/voc.names'
    elif m.num_classes == 80:
        namesfile = 'data/coco.names'
    else:
        namesfile = 'data/names'
    class_names = load_class_names(namesfile)
 
    use_cuda = 0
    if use_cuda:
        m.cuda()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    if not cap.isOpened():
        print("Unable to open camera")
        exit(-1) 

    while True:
        res, img = cap.read()
        if res:
            sized = cv2.resize(img, (m.width, m.height))
            bboxes = do_detect(m, sized, 0.45, 0.4, sharing.usegpu) #third value in this call sets the confidence needed to detect object?
            print('------')
            draw_img, waitsignal = plot_boxes_cv2(img, bboxes, None, class_names)
            cv2.namedWindow("Detections", cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty("Detections",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
            cv2.imshow("Detections", draw_img)
            if waitsignal == True:
                #cv2.waitKey(0)
                waitsignal = False
            #out.write(draw_img)
                  
            cv2.waitKey(1)
        else:
             print("Unable to read image")
             exit(-1)

############################################
if __name__ == '__main__':
    AlarmDetector.GlobeCreate()
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
    global counterimage
    counterimage = 0
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

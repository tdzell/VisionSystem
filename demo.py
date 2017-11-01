from utils import *
from darknet import Darknet
import cv2
import AlarmDetector
import pyueye


def demo(cfgfile, weightfile):
    #GlobeCreate()
    m = Darknet(cfgfile)
    m.print_network()
    m.load_weights(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))
    
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter('output.avi',fourcc,30.0,(640,480))
	
	
	
    if m.num_classes == 20:
        namesfile = 'data/voc.names'
    elif m.num_classes == 80:
        namesfile = 'data/coco.names'
    else:
        namesfile = 'data/names'
    class_names = load_class_names(namesfile)
 
    use_cuda = 1
    if use_cuda:
        m.cuda()

    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
    if not cap.isOpened():
        print("Unable to open camera")
        exit(-1) 

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

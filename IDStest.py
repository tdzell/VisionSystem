from pyueye import ueye
from pyueye_example_camera import Camera
from pyueye_example_utils import FrameThread
import cv2
import numpy as np

cam = Camera()
cam.init()
cam.set_colormode(ueye.IS_CM_BGR8_PACKED)
cam.set_aoi(0,0, 1280, 1024)
cam.alloc()
cam.capture_video()




thread = FrameThread(cam, view)
thread.start()






















'''
is_SetBinning
is_SetSubSampling
is_AOI
is_Exposure
is_SetFrameRate
is_EdgeEnhancement
is_Measure (blurriness detection)
is_CaptureVideo
is_ColorTemperature
'''
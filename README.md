












#######################Dependencies#######################

Python Set-Up things:

	-Anaconda [https://www.anaconda.com/download/ ] -- Python 3.6
	
	
	-pytorch [version 0.2.1 https://anaconda.org/peterjc123/pytorch/files ] -- install using conda, will probably need to be downloaded manually and installed using "offline" argument
	
	-torchvision [pip install torchvision?]
	
	-OpenCV [Windows: https://www.lfd.uci.edu/~gohlke/pythonlibs/; RasPi: https://www.pyimagesearch.com/2016/04/18/install-guide-raspberry-pi-3-raspbian-jessie-opencv-3/ ]
	
	-pyueye for IDS Machine Vision Camera [https://pypi.org/project/pyueye/#files ]
	
		-pyueye has it's own dependency that might have been "ctypes", but might have been something else





#######################LiveVideoDetection:#######################

python [program to run] [config file] [weights file] [GPU optimizaions] [IDS or Other]

ex: [python liveVideoDetect.py cfg/yolo-voc.cfg backup/000425.weights GPU Standard]

Arguments:

	config file:
		###the config file and weights file must both utilize the same YOLO model
		
		'cfg/yolo-voc.cfg' for 20 class full YOLO model
		'cfg/tiny-yolo-voc.cfg' for 20 class "tiny" YOLO model
		
	weights file:
		###the weights file and config file must both utilize the same YOLO model
		
		'backup/IDSRefined.weights' for "full YOLO" bolt detection model
		'yolo-voc.weights' for "full YOLO" PASCAL VOC detection model (detects people, etc)
		
		'backup_multidetect2/007800.weights' for "tiny YOLO" bolt detection model
		'tiny-yolo-voc.weights' for "tiny YOLO" PASCAL VOC detection model (detects people, etc)
		
	GPU optimizations:
		###the usage of GPU optimizations allows for much faster processing of each image, but is only possible on computers with Nvidia graphics cards
		
		'GPU' for a computer with a graphics card that supports CUDA
		'Standard' to not utilize GPU optimizations
		    ###any text in this position that isn't "GPU" defaults to the "Standard" option
			
	IDS or Other:
		###the IDS uEye machine vision camera requires additional code in order to pull images, which results in the need for this argument
		
		'IDS' if the IDS uEye machine vision camera is being used
		'Other' for a standard webcam (or to utilize the IDS camera without setting control)
		    ###any text in this position that isn't "IDS" defaults to the "Other" option
			
			
			
			
#######################Training#######################
			
python train.py [data file] [config file] [weights file]

ex: python train.py cfg/voc.data cfg/yolo-voc.cfg yolo-voc.weights

Arguments:

	config file:
		###the config file and weights file must both utilize the same YOLO model
		
		'cfg/yolo-voc.cfg' for 20 class full YOLO model
		'cfg/tiny-yolo-voc.cfg' for 20 class "tiny" YOLO model
		
	weights file:
		###the weights file and config file must both utilize the same YOLO model
		
		'yolo-voc.weights' to begin training from "full YOLO" PASCAL VOC detection model (detects people, etc)
		'tiny-yolo-voc.weights' to begin training "tiny YOLO" PASCAL VOC detection model (detects people, etc)
		'backup/XXXXXX.weights' (XXXXXX is most recent weights file) to begin training from a previous training checkpoint)

		
		
		
#######################FAQ#######################

How do I adjust how often a checkpoint is saved for a training?
	
	Adjust "save_interval" in train.py. Beware that the full yolo-voc model can be prone to crashing during training on a GPU due to lack of available VRAM.
	
How do I adjust when an "alarm code" is sent?
	
	AlarmDetector.py handles when an alarm code is set off, and what happens when the alarm code is set off. AlarmDetector is called from utils.py during plot_boxes_cv2.
	
How do I adjust what happens when an "alarm code" is sent?

	AlarmDetector.py handles when an alarm code is set off, and what happens when the alarm code is set off. AlarmDetector is called from utils.py during plot_boxes_cv2.


#######################Resources#######################

###General
	
	github for code as of time of writing: https://github.com/tdzell/VisionSystem
	
	YOLO neural network: https://pjreddie.com/darknet/yolo/
	
	IDS Machine Vision Camera: https://en.ids-imaging.com/home.html
	
	Webcam model as of time of writing: Microsoft LifeCam Studio
	
	PyTorch: http://pytorch.org/
	
	OpenCV: https://opencv.org/

	
### PyTorch on Windows
	
	PyTorch on Windows: https://anaconda.org/peterjc123/pytorch

	
### OpenCV on RasPi
	
	OpenCV on RasPi: https://www.pyimagesearch.com/2016/04/18/install-guide-raspberry-pi-3-raspbian-jessie-opencv-3/

	OpenCV optimizations on RasPi: https://www.pyimagesearch.com/2017/10/09/optimizing-opencv-on-the-raspberry-pi/

	
###Tensorflow
	
	Tensorflow: https://www.tensorflow.org/
	
	Google Object Detection API: https://github.com/tensorflow/models/tree/master/research/object_detection
	
	Compiling Tensorflow on Raspi: https://github.com/samjabrahams/tensorflow-on-raspberry-pi
	
	Real-time video implementation of Detection API: https://towardsdatascience.com/building-a-real-time-object-recognition-app-with-tensorflow-and-opencv-b7a2b4ebdc32

	
### Differences between Python 2 and 3:
	
	Python 2 uses / for division to integer without remainder, Python 3 implementation is //
	
	The format for the print function in Python 2 is different from the print function format in Python 3

	
### Useful Linux Commands:
	
	"cd": change directory, follow command with folder to move into
	
	"ls"/"dir": display contents of current directory
	
	"sudo shutdown now": shutdown computer (RasPi)
	
	"sudo reboot now": restart computer (RasPi)
	
	"source activate [enviroment]": begin using an already created Python enviroment -- the current RasPi enviroment for PyTorch is "torch", so "source activate torch"

	
### Useful Python Commands:
	
	"pip install [filename]" install a .whl file to add a Python library. If not connected to internet, all library dependencies will need to already be installed. Pip installs outside of a virtual enviroment are available to all enviroments.
	
	"pip install [library]" install a Python library using files from the internet. Pip installs outside of a virtual enviroment are available to all enviroments.
	
	"conda install [library]" install the Anaconda supported version of a Python library. Library will only be available from within an Anaconda enviroment.
	
	"conda install [filename] --offline" install the Anaconda supported version of a Python library without internet connection. All dependency libraries are automatically included in the file. Anaconda uses its own installation files seperate from .whl files.
	
	"python -m pip" useful to invoke a pip command if you are having problems invoking pip directly
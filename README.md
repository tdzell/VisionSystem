



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
		
		'backup/000425.weights' for "full YOLO" bolt detection model
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


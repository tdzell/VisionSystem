import os
import glob

for f in glob.glob('*.jpg'):
	new_name = f.replace('276853_D2768  ', "")
	os.rename(f,new_name)
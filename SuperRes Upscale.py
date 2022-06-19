# from https://towardsdatascience.com/deep-learning-based-super-resolution-with-opencv-4fd736678066
# https://github.com/Saafke/EDSR_Tensorflow/tree/master/models

import cv2
from cv2 import dnn_superres
import datetime, time

# Create an SR object
sr = dnn_superres.DnnSuperResImpl_create()

infile = input("Input file: ")
print("-Begin 4x Upscaling for " + infile)

# Start timing the process
time_begin = datetime.datetime.fromtimestamp(time.time())

# Read image
image = cv2.imread(infile)

# Read the desired model, in same folder as script
path = "EDSR_x4.pb"
sr.readModel(path)

# Set the desired model and scale to get correct pre- and post-processing
sr.setModel("edsr", 4)

# Upscale the image
result = sr.upsample(image)

# Save the image
cv2.imwrite((infile+"-upscaled.png"), result)

print("-EDSR Upscaling Complete")
print("-Result Written to Disk")

time_end = datetime.datetime.fromtimestamp(time.time())
print("Time elapsed: ", str(time_end - time_begin))
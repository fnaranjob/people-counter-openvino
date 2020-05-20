import numpy as np
import cv2

def process_input(image, height, width):
	processed_image = np.copy(image)
	processed_image = cv2.resize(processed_image,(width,height))
	processed_image = processed_image.transpose((2,0,1))
	processed_image = processed_image.reshape(1, 3, height, width)
	return processed_image

def process_output():
	return
import numpy as np
import cv2

def process_input(image, height, width):
	processed_image = np.copy(image)
	processed_image = cv2.resize(processed_image,(width,height))
	processed_image = processed_image.transpose((2,0,1))
	processed_image = processed_image.reshape(1, 3, height, width)
	return processed_image

def process_output(inference_output, threshold, input_width, input_height):
	person_detections = inference_output[inference_output[:,1]==1]
	person_detections = inference_output[inference_output[:,2]>=threshold]
	boxes=[]
	for detection in person_detections:
		pt1=(int(detection[3]*input_width), int(detection[4]*input_height))
		pt2=(int(detection[5]*input_width), int(detection[6]*input_height))
		box = {'pt1':pt1 , 'pt2':pt2}
		boxes.append(box)

	return boxes
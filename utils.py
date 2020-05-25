import numpy as np
import cv2
import os

#misc constants
BOX_COLORS = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(0,255,255),(255,0,255),(255,255,255),(128,0,0),(0,128,0),(0,0,128)]
LINE_THICKNESS = 2
FONT_THICKNESS = 1
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.5

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

def draw_results(image, boxes, people_in_frame, total_people_count, time_in_frame, average_time, inference_time):
	i=0
	if len(boxes)>=0:
		for box in boxes:
			cv2.rectangle(image,box['pt1'],box['pt2'],BOX_COLORS[i],LINE_THICKNESS)
			i=i+1

	if total_people_count!=-1: #video mode
		cv2.putText(image, "People in frame: %d"%people_in_frame, (20,30), FONT, FONT_SCALE, BOX_COLORS[0], FONT_THICKNESS)
		cv2.putText(image, "Total people counted: %d"%total_people_count, (20,55), FONT, FONT_SCALE, BOX_COLORS[0], FONT_THICKNESS)
		cv2.putText(image, "Current person time: %.1f [sec]"%time_in_frame, (20,80), FONT, FONT_SCALE, BOX_COLORS[0], FONT_THICKNESS)
		cv2.putText(image, "Average time in frame: %.1f [sec]"%average_time, (20,105), FONT, FONT_SCALE, BOX_COLORS[0], FONT_THICKNESS)
		cv2.putText(image, "Inference time: %d [msec]"%inference_time, (20,130), FONT, FONT_SCALE, BOX_COLORS[0], FONT_THICKNESS)

def validate_input(input_arg):
	single_image_mode=False
	if input_arg == 'CAM':
		input_validated = 0
	elif input_arg.endswith('.jpg') or input_arg.endswith('.bmp') :
		single_image_mode = True
		input_validated = input_arg
	else:
		input_validated = input_arg
		assert os.path.isfile(input_arg), "file doesn't exist"

	return input_validated, single_image_mode
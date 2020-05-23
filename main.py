"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import os
import sys
import time
import socket
import json
import cv2

import logging as log
import paho.mqtt.client as mqtt

from argparse import ArgumentParser
from inference import Network
import utils

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60

# Misc constants
SCALE=1 #needed for input 1 (image info vector)
FILTER_COUNT=5 #a person needs to remain detected for at least FILTER_COUNT frames to be considered present in the frame

def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    return parser


def connect_mqtt():

    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)
    client.loop_start()
    return client


def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    mqtt_client=connect_mqtt()
    infer_network = Network(args.model)
    prob_threshold = args.prob_threshold
    infer_network.load_model(args.device)
    n,c,h,w = infer_network.get_input_shape()
 
    cap=cv2.VideoCapture(args.input)
    if not cap.isOpened():
            exit("Error: couldn't open input file")
           
    input_width = int(cap.get(3))
    input_height = int(cap.get(4))
    frame_rate=cap.get(cv2.CAP_PROP_FPS)
    frame_count=0

    #stats vars
    current_people_before=0
    current_people_now=0
    current_people_buffer=0
    total_people_count=0
    time_in_frame=0.0 #time the current detected person has stayed so far [sec]
    total_times=[0.0] #list of total time in frame for all people detected so far
    average_time=0.0 #average time in frame for all people detected so far
    new_person_detected=False

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        start_time=time.time()
        processed_frame=utils.process_input(frame, h, w)
        input_dict=infer_network.get_inputs(processed_frame,h,w,SCALE)
        request_handle=infer_network.exec_inference(input_dict)
        infer_network.wait(request_handle)
        output=infer_network.get_output(request_handle)
        boxes=utils.process_output(output,args.prob_threshold,input_width,input_height)

        frame_count=frame_count+1        
        current_people_now=len(boxes)
        
        if (current_people_now != current_people_before):
            current_people_buffer=current_people_buffer+1
            new_person_detected=False

        if current_people_buffer == FILTER_COUNT:
            current_people_before = current_people_now
            current_people_buffer = 0

            if current_people_now != 0: #a new person was detected
                total_people_count=total_people_count+1
                mqtt_client.publish("person",json.dumps({"count": current_people_before}))
                #mqtt_client.publish("person",json.dumps({"total": total_people_count}))
                new_person_detected=True
            else: #no detections on frame anymore, store time person was in frame
                total_times.append(time_in_frame)
                mqtt_client.publish("person/duration",json.dumps({"duration": time_in_frame}))
                mqtt_client.publish("person",json.dumps({"count": 0}))
                average_time=sum(total_times)/total_people_count
                time_in_frame=0
        
        if(new_person_detected):
            time_in_frame = time_in_frame + 1/frame_rate

        inference_time=int((time.time()-start_time)*1000.0)
        utils.draw_results(frame, boxes, current_people_before, total_people_count, time_in_frame, average_time,inference_time)


        ### TODO: Send the frame to the FFMPEG server ###

        ### TODO: Write an output image if `single_image_mode` ###
    cap.release()
    client.loop_stop()
    client.disconnect()


def main():
    """
    Load the network and parse the output.

    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()
    # Connect to the MQTT server
    client = connect_mqtt()
    # Perform inference on the input stream
    infer_on_stream(args, client)


if __name__ == '__main__':
    main()

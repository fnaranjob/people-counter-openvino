#!/usr/bin/env python3
"""
 Copyright (c) 2018 Intel Corporation.

 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit persons to whom the Software is furnished to do so, subject to
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
import numpy as np
import os
import sys
import logging as log
from openvino.inference_engine import IENetwork, IECore


class Network:
    """
    Load and configure inference plugins for the specified target devices 
    and performs synchronous and asynchronous modes for the specified infer requests.
    Warning: To be used only with faster_rcnn_inception_v2 tensorflow model
    """

    IE=None
    net=None
    exec_net=None
    device=None
    input_dict=None #input dictionary ready for inference request

    def __init__(self, model_xml):
        self.IE=IECore()
        self.net=IENetwork(model=model_xml,weights=model_xml.replace('xml','bin'))
        self.net_inputs={}

    def __check_layers__(self):
        layers_map = self.IE.query_network(network=self.net,device_name=self.device)
        for layer in self.net.layers.keys():
            if layers_map.get(layer, "none") == "none":
                return False #Found unsupported layer
        return True
        

    def load_model(self,device_name):
        self.device=device_name
        if(self.__check_layers__()):
            self.exec_net=self.IE.load_network(network=self.net,device_name=device_name,num_requests=1)
        else:
            sys.exit("Unsupported model layer found, can't continue")

        #Using OpenVino V2020.1, no need for CPU extensions anymore

    def get_inputs(self, processed_image, height, width, scale):
        info_vec=np.array([[height,width,scale]]) #image info vector
        input_keys=[]
        for key in self.net.inputs:
            input_keys.append(key)
        input_dict={input_keys[0]:info_vec, input_keys[1]:processed_image}
        return input_dict

    def exec_inference(self):
        ### TODO: Start an asynchronous request ###
        ### TODO: Return any necessary information ###
        ### Note: You may need to update the function parameters. ###
        return

    def wait(self):
        ### TODO: Wait for the request to be complete. ###
        ### TODO: Return any necessary information ###
        ### Note: You may need to update the function parameters. ###
        return

    def get_output(self):
        ### TODO: Extract and return the output results
        ### Note: You may need to update the function parameters. ###
        return

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
    and perform asynchronous infer requests.
    Warning: To be used only with faster_rcnn_inception_v2 tensorflow model
    code expects model with 2 inputs: image_info and image_tensor
    output key to be used in the app is hardcoded
    """

    IE=None
    net=None
    exec_net=None
    device=None
    input_keys=[] #names of the model inputs for inference request

    def __init__(self, model_xml):
        self.IE=IECore()
        self.net=IENetwork(model=model_xml,weights=model_xml.replace('xml','bin'))
        for key in self.net.inputs:
            self.input_keys.append(key)

    def __check_layers__(self):
        good_to_go = True
        layers_map = self.IE.query_network(network=self.net,device_name=self.device)
        for layer in self.net.layers.keys():
            if layers_map.get(layer, "none") == "none":
                sys.stderr.write("Unsupported layer: "+layer+"\n")
                good_to_go=False
        return good_to_go
        

    def load_model(self,device_name):
        self.device=device_name
        if(self.__check_layers__()):
            self.exec_net=self.IE.load_network(network=self.net,device_name=device_name,num_requests=1)
        else:
            sys.exit("Unsupported layer found, can't continue")

        #Using OpenVino V2020.1, no need for CPU extensions anymore

    def get_input_shape(self):
        n,c,h,w = self.net.inputs[self.input_keys[1]].shape
        return n,c,h,w


    def get_inputs(self, processed_image, height, width, scale):
        #image info vector
        info_vec=np.array([[height,width,scale]]) 
        input_dict={self.input_keys[0]:info_vec, self.input_keys[1]:processed_image}
        return input_dict

    def exec_inference(self, input_dict):
        request_handle=self.exec_net.start_async(request_id=0, inputs=input_dict)
        return request_handle

    def wait(self, request_handle):
        return request_handle.wait()

    def get_output(self, request_handle):
        output=np.squeeze(request_handle.outputs["detection_output"])
        return output

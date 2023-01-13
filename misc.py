# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 11:06:10 2022

@author: quetscher
"""
import os
import csv
import pandas


    
    


import zmq

context = zmq.Context()

#  Socket to talk to server
print("Connecting to hello world server...")
socket = context.socket(zmq.REQ)
socket.connect("tcp://localhost:5555")
socket.send_string("export_mats: 54, -6, 231245324")
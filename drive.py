# connect model to server 


# required library 

import socketio								# realtime bi-directional communication between client and server 
import eventlet
import numpy as np
from flask import Flask							# microframework of python use to create webapp
from keras.models import load_model						# load model
import base64
from io import BytesIO
from PIL import Image
import cv2
import sys
import pickle

sio = socketio.Server()							# initalize web server 

app = Flask(__name__) 								#'__main__' creating instance of Flask					
speed_limit = 10								# max speed 
def img_preprocess(img):
    img = img[60:135,:,:]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img,  (3, 3), 0)
    img = cv2.resize(img, (200, 66))
    img = img/255
    return img


@sio.on('telemetry')								# eventhandler to listen the image send to model
def telemetry(sid, data):
    speed = float(data['speed'])						# 64 bit encoded
    image = Image.open(BytesIO(base64.b64decode(data['image'])))		# reading the decoded image captured using PIL library
    image = np.asarray(image)							# data to array
    image = img_preprocess(image)						# same preprocessing as in model
    image = np.array([image])							# again store to array
    steering_angle = float(model.predict(image))
    throttle = 1.0 - speed/speed_limit					# limiting speed
    print('{} {} {}'.format(steering_angle, throttle, speed))		# print angle , throttle and speed
    send_control(steering_angle, throttle)					# fun send control



@sio.on('connect')									# envent handler connect(reversed) , decorator
def connect(sid, environ):								
    print('Connected')									# print connect in terminal if the sio connec
    #send_control(0, 0)

def send_control(steering_angle, throttle):
    sio.emit('steer', data = {
        'steering_angle': steering_angle.__str__(),
        'throttle': throttle.__str__()
    })


if __name__ == '__main__':							
    model = load_model("model.h5") 
    app = socketio.Middleware(sio, app)						# middleware to dipatch trafic socketio servers with flask app
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)				# web server gateway interface (wsgi) to send requst to webserver, 												  listen- open listening socket 1st param - ip add, 2nd ip - port 

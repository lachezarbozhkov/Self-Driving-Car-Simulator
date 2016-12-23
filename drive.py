import argparse
import base64
from datetime import datetime
import json
import os

import numpy as np
import socketio
import eventlet
import eventlet.wsgi
import time
from PIL import Image
from PIL import ImageOps
from flask import Flask, render_template
from io import BytesIO

from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array
import cv2

sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None


img_path_exists = os.path.exists("simulator_imgs")
if img_path_exists:
    simulator_imgs = open(r'simulator_imgs/data.csv', 'a')


def save_img(image, steering_angle):
    if img_path_exists:
        file_path = datetime.now().strftime('simulator_imgs/%Y%m%d-%H%M%S-%f.jpg')
        # cv2.imwrite(file_path, cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        cv2.imwrite(file_path, image)
        simulator_imgs.write("{}, {}\n".format(file_path, steering_angle))


def load_img(img, resize=True, crop_top=20, crop_bottom=-1):
    if resize:
        img = cv2.resize(img, (160, 80), interpolation=cv2.INTER_CUBIC)
    if crop_top:
        img = img[crop_top:crop_bottom, :]

    return img


def ld_img(img):
    # return load_img(img, resize=True, crop_top=30, crop_bottom=-10)
    # return load_img(img, resize=False, crop_top=60, crop_bottom=-15)
    return load_img(img, resize=False, crop_top=60, crop_bottom=-1)


@sio.on('telemetry')
def telemetry(sid, data):
    # The current steering angle of the car
    steering_angle = data["steering_angle"]
    # The current throttle of the car
    throttle = data["throttle"]
    # The current speed of the car
    speed = data["speed"]
    # The current image from the center camera of the car
    imgString = data["image"]
    image = Image.open(BytesIO(base64.b64decode(imgString)))
    original_image = np.asarray(image)
    image_array = ld_img(original_image)
    transformed_image_array = image_array[None, :, :, :]
    # This model currently assumes that the features of the model are just the images. Feel free to change this.
    steering_angle = float(model.predict(transformed_image_array, batch_size=1))
    steering_angle = max(min(steering_angle, 1), -1)
    save_img(original_image, steering_angle)
    throttle = 0.2  # Use 0.3 for the second track
    print(steering_angle, throttle)
    send_control(steering_angle, throttle)


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit("steer", data={
    'steering_angle': steering_angle.__str__(),
    'throttle': throttle.__str__()
    }, skip_sid=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument('model', type=str,
    help='Path to model definition json. Model weights should be on the same path.')
    args = parser.parse_args()
    with open(args.model, 'r') as jfile:
        model = model_from_json(json.load(jfile))

    model.compile("adam", "mse")
    weights_file = args.model.replace('json', 'h5')
    model.load_weights(weights_file)

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)

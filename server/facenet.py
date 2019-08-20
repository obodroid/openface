#!/usr/bin/env python2
#
# Copyright 2015-2016 Carnegie Mellon University
# Edited 2018 Obodroid Corporation by Lertlove
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import ptvsd

# Allow other computers to attach to ptvsd at this IP address and port, using the secret
# ptvsd.enable_attach(address=('0.0.0.0', 3000))
# Pause the program until a remote debugger is attached
# ptvsd.wait_for_attach()

import os
import sys
fileDir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(fileDir, ".."))
import traceback
import dlib

import pickle
import pymongo
import pprint
import txaio
txaio.use_twisted()

from autobahn.twisted.websocket import WebSocketServerProtocol, \
    WebSocketServerFactory
from twisted.internet import task, reactor, threads
from twisted.internet.defer import Deferred, inlineCallbacks
from twisted.internet.ssl import DefaultOpenSSLContextFactory
from twisted.python import log

from face import Face
import headPoseEstimator as hp

import Queue
import threading
import urllib
import argparse
import cv2
import imagehash
import json
from PIL import Image
import numpy as np
import StringIO
import base64
import time
from datetime import datetime
import ssl
import scipy.misc

from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.svm import SVC
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.ensemble import IsolationForest
from sklearn.externals import joblib
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize

#import for use Face++
import requests
import json

import benchmark
import openface
import config

args = config.loadConfig()

class MidpointNormalize(Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))

class Facepp():
   def __init__(self, myKey, mySecret):
	   #request import requests
       self.myKey = myKey
       self.mySecret = mySecret
       self.path = "./../../data/imageType/"
       self.mapIndexHeadPoseDict = {'1': 'topLeft', '2': 'topMid', '3': 'topRight',
									'4': 'midLeft', '5': 'midMid', '6': 'midRight',
									'7': 'botLeft', '8': 'botMid', '9': 'botRight',
									'0': 'unknown'}

   def findMaxValueInDict(self, myDict):
   	   inverse = [(value, key) for key, value in myDict.items()]
   	   return max(inverse)[1]

   def detect(self, picBase64, face):
	   
       attributes="gender,age,ethnicity,mouthstatus,eyestatus,facequality,emotion,headpose"
	   
	   #url for contact with Face++ detect
       http_url = 'https://api-us.faceplusplus.com/facepp/v3/detect'
       try:
            #get values from Face++ 
            json_resp = requests.post(http_url,
                            data = { 
                                'api_key': self.myKey,
                                'api_secret': self.mySecret,
                                'image_base64': picBase64,
                                'return_attributes': attributes
                                }
                            )	   
            #transfrom json text to dictionary
            obj_content= json.loads(vars(json_resp)['_content'])
            dict_attributes = obj_content['faces'][0]['attributes']
            
            #age value
            face.ageFacepp = dict_attributes['age'].values()[0]
            #gender value (Male or Female)
            face.genderFacepp = dict_attributes['gender'].values()[0]
            #ethnicity value (Asian ,White, Black)
            face.ethnicityFacepp = dict_attributes['ethnicity'].values()[0]
            #emotion (anger ,disgust, fear, happiness, neutral, sadness, surprise)
            face.emotionFacepp = self.findMaxValueInDict(dict_attributes['emotion'])
            #mouth (surgical_mask_or_respirator, other_occlusion, close, open)
            face.mouthFacepp = self.findMaxValueInDict(dict_attributes['mouthstatus'])
            #eye status (occlusion, no_glass_eye_open, normal_glass_eye_close, normal_glass_eye_open, dark_glasses, no_glass_eye_close)
            face.lefteyeStatusFacepp = self.findMaxValueInDict(dict_attributes['eyestatus']['left_eye_status'])
            face.righteyeStatusFacepp = self.findMaxValueInDict(dict_attributes['eyestatus']['right_eye_status'])
            #if threshold is less than values, this picture can comparable.
            faceth = self.findMaxValueInDict(dict_attributes['facequality'])
            if faceth == 'value':
                face.facequalityFacepp = 'high'
            else:
                face.facequalityFacepp = 'low'
            face.headposeFacepp = dict_attributes['headpose']
       except:
            face.ageFacepp = None
            face.genderFacepp = None
            face.ethnicityFacepp = None
            face.emotionFacepp = None
            face.mouthFacepp = None
            face.lefteyeStatusFacepp = None
            face.righteyeStatusFacepp = None
            face.facequalityFacepp = None
            face.headposeFacepp = None
            pass

   def save9typeOfFace(self, nameImage, inputImage, indexFace):
		# index = 1 (top-left), index = 2 (top-mid), index = 3 (top-right)
		# index = 4 (mid-left), index = 5 (mid-mid), index = 6 (mid-right)
		# index = 7 (bottom-left), index = 8 (bottom-mid), index = 9 (bottom-right)

		mapIndexFaceandFloderDict = {'1': 'topLeft', '2': 'topMid', '3': 'topRight',
									'4': 'midLeft', '5': 'midMid', '6': 'midRight',
									'7': 'botLeft', '8': 'botMid', '9': 'botRight',
									'0': 'unknown'}
		cv2.imwrite(os.path.join(self.path+mapIndexFaceandFloderDict[indexFace] , nameImage), inputImage)

   def found9typeOfFace(self, face):
       	# index = 1 (top-left), index = 2 (top-mid), index = 3 (top-right)
		# index = 4 (mid-left), index = 5 (mid-mid), index = 6 (mid-right)
		# index = 7 (bottom-left), index = 8 (bottom-mid), index = 9 (bottom-right)
	#   global indexFace
       if face.headposeFacepp != None:
            if face.headposeFacepp['yaw_angle'] > 14:
                if face.headposeFacepp['pitch_angle'] < -9:
                    indexFace = '1'
                elif face.headposeFacepp['yaw_angle'] > 40 and face.headposeFacepp['pitch_angle'] > 0 and face.headposeFacepp['pitch_angle'] < 10 and abs(face.headposeFacepp['roll_angle']) < 10:
                    indexFace = '4'
                elif face.headposeFacepp['yaw_angle'] > 40 and abs(face.headposeFacepp['pitch_angle']) < 5 and face.headposeFacepp['roll_angle'] < -18.5:
					indexFace = '7'
                elif face.headposeFacepp['yaw_angle'] > 25 and face.headposeFacepp['pitch_angle']<2 and face.headposeFacepp['pitch_angle']> -9 and face.headposeFacepp['roll_angle'] >= 1:
                    indexFace = '1'
                elif face.headposeFacepp['pitch_angle'] > 10:
                    indexFace = '7'
                else:
                    indexFace = '4'
            elif face.headposeFacepp['yaw_angle'] < -11.5:
                if face.headposeFacepp['pitch_angle'] < -7:
                    indexFace = '3'
                elif face.headposeFacepp['pitch_angle'] < 12 and face.headposeFacepp['pitch_angle'] > 6.5 and abs(face.headposeFacepp['roll_angle']) < 20:
                    indexFace = '6'
                elif face.headposeFacepp['pitch_angle'] > 6.5:
                    indexFace = '9'
                elif face.headposeFacepp['yaw_angle'] < -25 and face.headposeFacepp['pitch_angle'] < 0 and face.headposeFacepp['pitch_angle'] >-9 and face.headposeFacepp['roll_angle'] < -13:
                    indexFace = '3'
                elif face.headposeFacepp['yaw_angle'] < -25 and face.headposeFacepp['pitch_angle'] < 12 and face.headposeFacepp['pitch_angle'] >0 and face.headposeFacepp['roll_angle'] < -9:
                    indexFace = '3'
                elif face.headposeFacepp['pitch_angle'] > 0 and face.headposeFacepp['pitch_angle'] <9 and face.headposeFacepp['roll_angle'] > 11:
                    indexFace = '9'
                else:
                    indexFace = '6'
            elif face.headposeFacepp['yaw_angle'] <= 15 and face.headposeFacepp['yaw_angle'] >= -11.5:
                if face.headposeFacepp['pitch_angle'] < -9:
                    indexFace = '2'
                elif face.headposeFacepp['pitch_angle'] > 10:
                    indexFace = '8'
                elif abs(face.headposeFacepp['roll_angle']) <= 10:
                    indexFace = '5'
            else:
                indexFace = '0'
       else:
            indexFace = '0'
        
       return indexFace



class Facenet():
    def __init__(self, workerIndex):
        self.workerIndex = workerIndex
        self.images = {}
        self.detectQueue = Queue.Queue(maxsize=10)
        self.detectWorker = threading.Thread(target=self.consume)
        self.detectWorker.setDaemon(True)
        self.detectWorker.start()
        self.recentFaces = []
        self.recentPeople = {}
        self.processRecentFace = False
        self.enableClassifier = True
        self.training = True
        self.lastLogTime = time.time()

        self.sp = dlib.shape_predictor(args.shapePredictor)
        self.hpp = dlib.shape_predictor(args.headPosePredictor)
        self.eye_cascade = cv2.CascadeClassifier(args.eyeCascade)

        if args.facePredictor:
            self.cnn_face_detector = dlib.cnn_face_detection_model_v1(args.facePredictor)
        else:
            self.hog_detector = dlib.get_frontal_face_detector()

        if args.faceRecognitionModel:
            self.fr_model = dlib.face_recognition_model_v1(args.faceRecognitionModel)
        else:
            self.fr_model = None

    def qput(self, msg, callback):
        print("detectQueue qsize: {}".format(self.detectQueue.qsize()))
        benchmark.startAvg(10.0, "dropframe")
        if self.detectQueue.full():
            self.detectQueue.get()
            benchmark.updateAvg("dropframe")
        self.detectQueue.put([msg, callback])

    def consume(self):
        # TODO need flag to stop benchmark
        while True:
            if not self.detectQueue.empty():
                msg, callback = self.detectQueue.get()
                print("facenetWorker-{}  consume : {}".format(self.workerIndex, msg['keyframe']))
                self.processFrame(msg, callback)
            time.sleep(0.001)
        
    def processFrame(self, msg, callback):
        try:
            start = time.time()
            dataURL = msg['dataURL']

            if msg.has_key("keyframe"):
                keyframe = msg['keyframe']
            else:
                keyframe = start

            if msg.has_key("robotId"):
                robotId = msg['robotId']
            else:
                robotId = ""

            if msg.has_key("videoId"):
                videoId = msg['videoId']
            else:
                videoId = ""
            
            videoSerial = "{}-{}".format(robotId, videoId)
            frameSerial = "{}_{}".format(videoSerial, keyframe)

            if msg.has_key("label"):
                label = msg['label']
            else:
                label = None

            if msg.has_key("bbox"):
                bbox = msg['bbox']
            else:
                bbox = None

            self.logProcessTime(
                "0_start", "Start processing frame {}".format(frameSerial), robotId, videoId, keyframe)

            # construct numpy array of PIL image from base64 str
            head = "data:image/jpeg;base64,"
            assert(dataURL.startswith(head))
            imgData = base64.b64decode(dataURL[len(head):])
            imgStr = StringIO.StringIO()
            imgStr.write(imgData)
            imgStr.seek(0)
            imgPIL = Image.open(imgStr)

            self.logProcessTime(
                "1_open_image", "Open PIL Image from base64 {}".format(frameSerial), robotId, videoId, keyframe)

            npImg = np.asarray(imgPIL)

            # save input image if you want to compare later
            if args.saveImg:
                imgPIL.save(os.path.join(args.imgPath, 'input',
                                         '{}.jpg'.format(frameSerial)))
                self.logProcessTime(
                    "2_save_image", "Save input image {}".format(frameSerial), robotId, videoId, keyframe)

            # detect face with cnn of hog
            if args.facePredictor:
                benchmark.start(
                    "cnn_face_detector_{}".format(frameSerial))
                bbs = self.cnn_face_detector(npImg, 0)
                benchmark.update(
                    "cnn_face_detector_{}".format(frameSerial))
                tag, elasped, rate = benchmark.end(
                    "cnn_face_detector_{}".format(frameSerial))
                if rate:
                    if len(bbs) > 0:
                        benchmark.logInfo("{} facePredictor_found : {:.2f}, {:.4f}, {}, {}".format(
                            tag, rate, elasped, npImg.shape, len(bbs)))
                    else:
                        benchmark.logInfo("{} facePredictor_not_found : {:.2f}, {:.4f}, {}, {}".format(
                            tag, rate, elasped, npImg.shape, len(bbs)))
            else:
                benchmark.start("hog_detector_{}".format(frameSerial))
                bbs = self.hog_detector(npImg, 1)
                benchmark.update("hog_detector_{}".format(frameSerial))
                benchmark.end("hog_detector_{}".format(frameSerial))

            print("Number of faces detected: {}".format(len(bbs)))
            self.logProcessTime(
                "3_face_detected", "Detector get face bounding box {}".format(frameSerial), robotId, videoId, keyframe)

            # iterate all detected faces to check upon conditions
            for index, bb in enumerate(bbs):
                # rule-1: Drop >>> low face detection confidence
                if args.facePredictor:
                    print("Face detection confidence = {}".format(bb.confidence))
                    if bb.confidence < 0.8:
                        print("Drop low confidence face detection")
                        continue
                    bb = bb.rect

                # rule-2: Drop >>> if face image dimension corrupted
                if ((bb.left() < 0) | (bb.right() < 0) | (bb.top() < 0) | (bb.bottom() < 0)):
                    print("rule-2")
                    continue

                if bbox is not None:
                    bbox['x'] += bb.left()
                    bbox['y'] += bb.top()
                    bbox['w'] = bb.width()
                    bbox['h'] = bb.height()

                # cropped face image
                cropImg = npImg[bb.top():bb.bottom(),
                              bb.left():bb.right()]
                _, jpgImg = cv2.imencode(
                    '.jpg', cv2.cvtColor(cropImg, cv2.COLOR_RGB2BGR))
                content = base64.b64encode(jpgImg) # Create base64 cropped image
                self.logProcessTime(
                    "4_crop_image", 'Crop image', robotId, videoId, keyframe)

                phash = str(imagehash.phash(Image.fromarray(cropImg)))

                # change to gray image to check blurry and headpose
                grayImg = cv2.cvtColor(npImg, cv2.COLOR_RGB2GRAY)
                cropGrayImg = cv2.cvtColor(cropImg, cv2.COLOR_RGB2GRAY)

                laplacianImg = cv2.Laplacian(cropGrayImg, cv2.CV_64F)
                focus_measure = laplacianImg.var()

                # rule-3: drop blurry face
                print("Focus Measure: {}".format(focus_measure))
                blur = focus_measure < args.focusMeasure

                if args.saveImg:
                    laplacianImgPIL = scipy.misc.toimage(laplacianImg)
                    laplacianImgPIL.save(os.path.join(
                        args.imgPath, 'output', 'laplacian_{}-{}_{}-{}_{}_{}.png'.format('blur' if blur else 'sharp', focus_measure, robotId, videoId, keyframe, index + 1)))
                    cropImgPIL = scipy.misc.toimage(cropImg)
                    cropImgPIL.save(os.path.join(
                        args.imgPath, 'output', '{}-{}_{}-{}_{}_{}.png'.format('blur' if blur else 'sharp', focus_measure, robotId, videoId, keyframe, index + 1)))
                    self.logProcessTime(
                        "5_save_crop_image", 'Save Cropped image output', robotId, videoId, keyframe)

                if blur:
                    print("Drop blurry face")
                    continue

                foundFace = Face(None, phash=phash, content=content, label=label, bbox=bbox)

                #request data from facepp        
                facepp1 = Facepp(args.key_APIFace, args.secret_APIFace)
                facepp1.detect(dataURL, foundFace)
                indexFace = facepp1.found9typeOfFace(foundFace)
                print( 'indexFace = {}'.format(indexFace) )
                sideFace = False
                if foundFace.headposeFacepp != None:
                    if indexFace == '5':
                            sideFace = False
                    else:
                            sideFace = True
                print("sideFace: {}".format(sideFace))

                # rule-4: FOUND (detect) >>> low resolution face
                if bb.width() < args.minFaceResolution or bb.height() < args.minFaceResolution:
                    callback(robotId, videoId, keyframe, foundFace)
                    return 

                # # rule-5: FOUND (detect) >>> side face
                # headPose = self.hpp(grayImg, bb)
                # headPoseImage, p1, p2 = hp.pose_estimate(grayImg, headPose)
                # headPoseLength = cv2.norm(
                #     np.array(p1) - np.array(p2)) / bb.width() * 100
                # print("Head Pose Length: {}".format(headPoseLength))

                # cropGrayImg = headPoseImage[bb.top():bb.bottom(),
                #                             bb.left():bb.right()]
                # sideFace = headPoseLength > args.sideFaceThreshold
                # print("sideFace: {}".format(sideFace))

                # if args.NUM_WORKERS == 1:
                #     eyes = self.eye_cascade.detectMultiScale(cropGrayImg)
                #     for (ex, ey, ew, eh) in eyes:
                #         cv2.rectangle(cropGrayImg, (ex, ey),
                #                       (ex+ew, ey+eh), (0, 255, 0), 2)
                #     cv2.putText(cropGrayImg, 'Side' if sideFace else 'Front',
                #                 (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))
                #     cv2.imshow('Head Pose', cropGrayImg)
                #     cv2.waitKey(1)
                #     if args.saveImg:
                #         cv2.imwrite(
                #             "images/side_"+datetime.now().strftime("%Y-%m-%d_%H:%M:%S.%f")+".jpg", cropGrayImg)
                
                if sideFace:
                    print("found non-frontal face")
                    callback(robotId, videoId, keyframe, foundFace)
                    return

                # get face descriptor or representations from face recogntion model
                if args.faceRecognitionModel:
                    benchmark.start("compute_face_descriptor_{}_{}".format(
                        frameSerial, index))
                    shape = self.sp(npImg, bb)
                    foundFace.rep = np.array(
                        self.fr_model.compute_face_descriptor(npImg, shape))
                    benchmark.update("compute_face_descriptor_{}_{}".format(
                        frameSerial, index))
                    benchmark.end("compute_face_descriptor_{}_{}".format(
                        frameSerial, index))

                self.logProcessTime(
                    "6_feed_network", 'Neural network forward pass', robotId, videoId, keyframe)

                callback(robotId, videoId, keyframe, foundFace)

            print("Finished processing frame {} for {} seconds.".format(
                keyframe, time.time() - start))
        except:
            print(traceback.format_exc())

    def logProcessTime(self, step, logMessage, robotId, videoId, keyframe):
        pass

numWorkers = args.NUM_WORKERS
numGpus = args.NUM_GPUS
facenetWorkers = []
print("facenet numWorkers : {}".format(numWorkers))

loadIndex = 0

for i in range(numWorkers):
    facenetWorkers.append(Facenet(i))

def putLoad(msg, callback):
    global loadIndex
    print("putLoad loadIndex = {}".format(loadIndex))
    facenetWorkers[loadIndex % numWorkers].qput(msg, callback)
    loadIndex = loadIndex + 1
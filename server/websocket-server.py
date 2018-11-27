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

import benchmark
import openface

modelDir = os.path.join(fileDir, '..', 'models')
dlibModelDir = os.path.join(modelDir, 'dlib')
openfaceModelDir = os.path.join(modelDir, 'openface')

# For TLS connections
tls_crt = os.path.join(fileDir, 'tls', 'server.crt')
tls_key = os.path.join(fileDir, 'tls', 'server.key')

parser = argparse.ArgumentParser()
parser.add_argument('--shapePredictor', type=str, help="Path to dlib's shape predictor.",
                    default=os.path.join(dlibModelDir, "shape_predictor_5_face_landmarks.dat"))
parser.add_argument('--headPosePredictor', type=str, help="Path to dlib's shape predictor.",
                    default=os.path.join(dlibModelDir, "shape_predictor_68_face_landmarks.dat"))
parser.add_argument('--facePredictor', type=str, help="Path to dlib's cnn face predictor.",
                    default=os.path.join(dlibModelDir, "mmod_human_face_detector.dat"))
parser.add_argument('--faceRecognitionModel', type=str, help="Path to dlib's face recognition model.",
                    default=os.path.join(dlibModelDir, 'dlib_face_recognition_resnet_model_v1.dat'))
parser.add_argument('--eyeCascade', type=str, help="Path to eye cascade.",
                    default=os.path.join(modelDir, "haarcascade_eye.xml"))
parser.add_argument('--imgDim', type=int,
                    help="Default image dimension.", default=96)
parser.add_argument('--imgPath', type=str, help="Path to images.",
                    default=os.path.join(fileDir, '..', 'data'))
parser.add_argument('--cuda', action='store_true')
parser.add_argument('--verbose', action='store_true')
parser.add_argument('--saveImg', action='store_true')
parser.add_argument('--port', type=int, default=9000,
                    help='WebSocket Port')
parser.add_argument('--recentFaceTimeout', type=int,
                    help="Recent face timeout", default=10)
parser.add_argument('--maxThreadPoolSize', type=int,
                    help="Max thread pool size", default=10)
parser.add_argument('--dth', type=str,
                    help="Representation distance threshold for recent face", default=0.2)
parser.add_argument('--minFaceResolution', type=int,
                    help="Minimum face area resolution", default=150)
parser.add_argument('--loosenFactor', type=float,
                    help="Factor used to loosen classifier neighboring distance", default=1.4)
parser.add_argument('--focusMeasure', type=int,
                    help="Threshold for filtering out blurry image", default=20)
parser.add_argument('--sideFaceThreshold', type=int,
                    help="Threshold for filtering out side face image", default=8)
parser.add_argument('--confidenceThreshold', type=float,
                    help="Threshold for filtering out unconfident face classification", default=0.2)
parser.add_argument('--classifier', type=str,
                    choices=['SVC',
                             'RadiusNeighbors'],
                    help='The type of classifier to use.',
                    default='RadiusNeighbors')

args = parser.parse_args()
sp = dlib.shape_predictor(args.shapePredictor)
hpp = dlib.shape_predictor(args.headPosePredictor)
eye_cascade = cv2.CascadeClassifier(args.eyeCascade)

if args.facePredictor:
    cnn_face_detector = dlib.cnn_face_detection_model_v1(args.facePredictor)
else:
    hog_detector = dlib.get_frontal_face_detector()

if args.faceRecognitionModel:
    fr_model = dlib.face_recognition_model_v1(args.faceRecognitionModel)
else:
    fr_model = None


class MidpointNormalize(Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


class OpenFaceServerProtocol(WebSocketServerProtocol):
    def __init__(self):
        super(OpenFaceServerProtocol, self).__init__()
        self.images = {}
        self.recentFaces = []
        self.recentPeople = {}
        self.processRecentFace = False
        self.enableClassifier = True
        self.training = True
        self.lastLogTime = time.time()

        self.modelFile = "working_classifier.pkl"
        (self.people, self.classifier) = self.getPreTrainedModel(self.modelFile)

        self.datasetFile = "working_dataset.pkl"
        (self.trainingData, self.calibrationSet) = self.getPreTrainedDataset(
            self.datasetFile)

        self.tsneFile = "working_tsne.pkl"
        self.tsneData = self.getPreTrainedTSNE(self.tsneFile)

        self.le = LabelEncoder().fit(self.people.keys())

        self.unknowns = {}
        self.faceId = 1
        self.processCount = 0

    def getPreTrainedModel(self, filename):
        if os.path.isfile(filename):
            return joblib.load(filename)
        return (dict(), None)

    def getPreTrainedDataset(self, filename):
        if os.path.isfile(filename):
            return joblib.load(filename)
        return ([], None)

    def getPreTrainedTSNE(self, filename):
        if os.path.isfile(filename):
            return joblib.load(filename)
        return None      

    def onConnect(self, request):
        print("Client connecting: {0}".format(request.peer))
        self.training = True

    def onOpen(self):
        print("WebSocket connection open.")

    def onMessage(self, payload, isBinary):
        raw = payload.decode('utf8')
        msg = json.loads(raw)

        if msg['type'] == "SETUP":
            print("\n on message: SETUP \n")
            if msg['debug']:
                benchmark.enable = True
            return

        if msg['type'] == "ALL_STATE":
            self.loadState(msg['images'], msg['training'], msg['people'])
        elif msg['type'] == "NULL":
            self.sendMessage('{"type": "NULL"}')
        elif msg['type'] == "FRAME":
            print("\n on message: FRAME \n")
            if args.maxThreadPoolSize == 1:
                self.processFrame(msg)
                return

            from datetime import datetime
            from time import sleep

            def mockStartThread():  # used for increasing thread pool size
                sleep(5)
            if len(reactor.getThreadPool().threads) < args.maxThreadPoolSize:
                reactor.callLater(
                    0, lambda: reactor.callInThread(mockStartThread))

            now = datetime.now()
            time_diff = now - \
                datetime.strptime(msg['time'], '%Y-%m-%dT%H:%M:%S.%fZ')
            print("frame latency: {}".format(time_diff))
            if time_diff.seconds < 1:
                reactor.callLater(
                    0, lambda: reactor.callInThread(self.processFrame, msg))
            else:
                print("drop delayed frame")
        elif msg['type'] == "PROCESS_RECENT_FACE":
            print("process recent face: {}".format(msg['val']))
            self.processRecentFace = msg['val']
        elif msg['type'] == "ENABLE_CLASSIFIER":
            print("enable classifier: {}".format(msg['val']))
            self.enableClassifier = msg['val']
        elif msg['type'] == "TRAINING":
            self.training = msg['val']
            if not self.training:
                self.trainClassifier()
        elif msg['type'] == 'SET_MAX_FACE_ID':
            self.faceId = int(msg['val']) + 1
        elif msg['type'] == "REQ_SYNC_IDENTITY":
            def getPeople(peopleId, label): return {
                'peopleId': peopleId,
                'label': label
            }
            newMsg = {
                "type": "SYNC_IDENTITY",
                "people": map(getPeople, self.people.keys(), self.people.values())
            }
            self.sendMessage(json.dumps(newMsg))
        elif msg['type'] == "UPDATE_IDENTITY":
            h = msg['hash'].encode('ascii', 'ignore')
            if h in self.images:
                self.images[h].identity = msg['idx']
                if not self.training:
                    self.trainClassifier()
            else:
                print("Image not found.")
        elif msg['type'] == "REMOVE_IMAGE":
            h = msg['hash'].encode('ascii', 'ignore')
            if h in self.images:
                del self.images[h]
                if not self.training:
                    self.trainClassifier()
            else:
                print("Image not found.")
        elif msg['type'] == 'REQ_TSNE':
            self.sendTSNE(msg['people'] if 'people' in msg else self.people)
        elif msg['type'] == 'CLASSIFY':
            self.classifyFace(np.array(msg['rep']))
        else:
            print("Warning: Unknown message type: {}".format(msg['type']))

    def onClose(self, wasClean, code, reason):
        print("WebSocket connection closed: {0}".format(reason))

    def loadState(self, jsImages, training, jsPeople):
        self.training = training
        self.images = {}

        for jsImage in jsImages:
            h = jsImage['hash'].encode('ascii', 'ignore')
            self.images[h] = Face(np.array(jsImage['rep']),
                                  jsImage['identity'])

        label_ids = [int(o['people_id']) for o in jsPeople]
        labels = [str(o['label']) for o in jsPeople]
        self.people = dict(zip(label_ids, labels))
        self.le = LabelEncoder().fit(self.people.keys())

        if not training:
            self.trainClassifier()

    def getData(self):
        X = []
        y = []

        for img in self.images.values():
            X.append(img.rep)
            y.append(self.le.transform([img.identity])[0])

        numIdentities = len(set(y + [-1])) - 1
        print("numIdentities = {}, numClasses = {}".format(
            numIdentities, len(self.people)))

        if len(self.people) < 2:
            print("Number of classes must be at least 2")
            return None

        if numIdentities < len(self.people):
            print("No image for {} classes".format(
                len(self.people) - numIdentities))
            return None

        X = np.vstack(X)
        y = np.array(y)
        return (X, y)

    def sendTSNE(self, people):
        if self.tsneData is None:
            d = self.getData()
            if d is not None:
                (X, y) = d

                print("TSNE fit transform")
                nc = None if len(X) < 50 else 50
                p_div = 7  # TODO : implement automatic perplexity selection algorithm
                p = len(X) / p_div if len(X) < 30 * p_div else 30
                p = 2 if p < 2 else p
                X_pca = PCA(n_components=nc).fit_transform(X, X)
                tsne = TSNE(n_components=2, n_iter=10000, random_state=0, perplexity=p)
                X_r = tsne.fit_transform(X_pca)

                print("Label encoder inverse transform")
                label_ids = self.le.inverse_transform(y)

                def getDataPoint(labelId, value, phash): return {
                    'labelId': labelId,
                    'value': value,
                    'phash': phash
                }

                self.tsneData = map(getDataPoint, label_ids, X_r.tolist(), self.images.keys())

                print("Saving working tsne to '{}'".format(self.tsneFile))
                joblib.dump(self.tsneData, self.tsneFile)

        msg = {
            "type": "TSNE_DATA",
            "data": self.tsneData
        }

        self.sendMessage(json.dumps(msg))

    def trainClassifier(self):
        self.tsneData = None
        d = self.getData()
        if d is None:
            self.classifier = None
            return
        else:
            (X, y) = d
            self.calibrationSet = []

            print("Training Classifier on {} labeled images.".format(
                len(self.images)))

            test_size = float(len(self.people)) / len(self.images) if len(
                self.images) * 0.1 < len(self.people) else 0.1

            cv = StratifiedShuffleSplit(
                n_splits=3, test_size=test_size, random_state=0)

            if args.classifier == 'RadiusNeighbors':
                radius_range = np.linspace(0.1, 1.5, num=15)
                param_grid = dict(radius=radius_range)
                grid = GridSearchCV(RadiusNeighborsClassifier(
                    outlier_label=-1), param_grid=param_grid, cv=cv, n_jobs=4).fit(X, y)

                scores = grid.cv_results_[
                    'mean_test_score'].reshape(len(radius_range), 1)

                print("The best parameters are %s with a score of %0.2f"
                      % (grid.best_params_, grid.best_score_))

                X_train, X_calibration, y_train, y_calibration = train_test_split(
                    X, y, test_size=test_size, random_state=0)

                # use all data as training data
                X_train, y_train = X, y

                loosen_factor = args.loosenFactor
                self.classifier = RadiusNeighborsClassifier(radius=grid.best_params_[
                                                            'radius'] * loosen_factor, weights='distance', outlier_label=-1, n_jobs=-1).fit(X_train, y_train)

                print("Train classifier with loosen radius of {}".format(
                    grid.best_params_['radius'] * loosen_factor))

                neighbors = self.classifier.radius_neighbors(
                    X_calibration, return_distance=False)

                for i, neighbor in enumerate(neighbors):
                    if neighbor.shape[0] > 0:
                        X_neighbor = np.take(X_train, neighbor, axis=0)
                        y_neighbor = np.full(
                            X_neighbor.shape[0], y_calibration[i])
                        self.calibrationSet.append(
                            round((1 - self.classifier.score(X_neighbor, y_neighbor)) * X_neighbor.shape[0]))

                self.trainingData = (X_train, y_train)

            elif args.classifier == 'SVC':
                C_range = np.logspace(-2, 7, 10)
                gamma_range = np.logspace(-6, 3, 10)
                param_grid = dict(gamma=gamma_range, C=C_range)
                grid = GridSearchCV(SVC(kernel='rbf'),
                                    param_grid=param_grid, cv=cv).fit(X, y)
                scores = grid.cv_results_['mean_test_score'].reshape(
                    len(C_range), len(gamma_range))

                print("The best parameters are %s with a score of %0.2f"
                      % (grid.best_params_, grid.best_score_))

                self.classifier = SVC(C=grid.best_params_['C'], kernel='rbf',
                                      probability=True, gamma=grid.best_params_['gamma']).fit(X, y)

                self.trainingData = (X, y)

            print("Saving working classifier to '{}'".format(self.modelFile))
            joblib.dump((self.people, self.classifier), self.modelFile)

            print("Saving working dataset to '{}'".format(self.datasetFile))
            joblib.dump((self.trainingData, self.calibrationSet),
                        self.datasetFile)

        msg = {
            "type": "TRAINING_FINISHED"
        }
        self.sendMessage(json.dumps(msg))

        if args.verbose:
            if args.classifier == 'RadiusNeighbors':
                self.visualizeParamHeatMap(scores, radius_range, [0])
            elif args.classifier == 'SVC':
                self.visualizeParamHeatMap(scores, C_range, gamma_range)

    def visualizeParamHeatMap(self, scores, param1, param2):
        plt.figure(figsize=(8, 6))
        plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
        plt.imshow(scores, interpolation='nearest', cmap=plt.cm.hot,
                   norm=MidpointNormalize(vmin=0.2, midpoint=0.92))
        plt.xlabel('param2')
        plt.ylabel('param1')
        plt.colorbar()
        plt.xticks(np.arange(len(param2)), param2)
        plt.yticks(np.arange(len(param1)), param1)
        plt.title('Validation Accuracy')
        plt.show()

    def getRecentFace(self, rep):
        recentFaceId = None

        def isValid(face):
            timeDiff = datetime.now() - face['time']
            return timeDiff.total_seconds() < args.recentFaceTimeout

        self.recentFaces = filter(isValid, self.recentFaces)

        for recentFace in self.recentFaces:
            d = rep - recentFace['rep']
            drep = np.dot(d, d)

            if args.verbose:
                print(
                    "Squared l2 distance between representations: {:0.3f}".format(drep))

            if drep < args.dth:
                print("recent face found")
                recentFaceId = recentFace['faceId']
                break

        if recentFaceId is None:
            self.recentFaces.append({
                'faceId': self.faceId,
                'rep': rep,
                'time': datetime.now()
            })

        return recentFaceId

    def createUnknownFace(self, rep, cluster, phash, content):
        face = Face(rep, None, cluster, phash, content)
        self.unknowns[phash] = face
        return face

    def foundUser(self, robotId, videoId, keyframe, face):
        print("found people id: {}".format(face.identity))

        msg = {
            "type": "FOUND_USER",
            "robotId": robotId,
            "videoId": videoId,
            "keyframe": keyframe,
            "phash": face.phash,
            "content": face.content,
            "rep": face.rep.tolist() if face.rep is not None else None,
            "time": datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'),
        }

        if face.cluster:
            msg["predict_face_id"] = face.cluster

        if face.cluster == self.faceId:
            self.faceId += 1

        if face.label:
            msg["label"] = face.label
        elif face.identity:
            msg["predict_name"] = face.label
            msg["predict_people_id"] = face.identity

        self.sendMessage(json.dumps(msg))

    def classifyFace(self, rep):
        peopleId, label, confidence, neighborPeopleIds, neighborDistances = None, None, None, None, None

        if self.classifier:
            if isinstance(self.classifier, SVC):
                predictions = self.classifier.predict_proba(
                    rep.reshape(1, -1)).ravel()
                predictIndex = np.argmax(predictions)
                peopleId = self.le.inverse_transform(predictIndex)
                person = self.people[peopleId]
                label = person.decode('utf-8')
                confidence = predictions[predictIndex]

            elif isinstance(self.classifier, RadiusNeighborsClassifier):
                neighbors = self.classifier.radius_neighbors(
                    [rep], return_distance=True)

                neighborDistances = neighbors[0][0]
                neighborIndices = np.take(
                    self.trainingData[1], neighbors[1][0], axis=0)
                neighborPeopleIds = self.le.inverse_transform(neighborIndices)
                print("\nNearest neighbor peopleIds : {}".format(neighborPeopleIds))

                predictIndex = self.classifier.predict(rep.reshape(1, -1))[0]
                if predictIndex >= 0:
                    peopleId = self.le.inverse_transform(predictIndex)
                    person = self.people[peopleId]
                    label = person.decode('utf-8')

                    X_neighbor = np.take(
                        self.trainingData[0], neighbors[1][0], axis=0)
                    y_neighbor = np.full(X_neighbor.shape[0], predictIndex)
                    nonconformity = (
                        1 - self.classifier.score(X_neighbor, y_neighbor)) * X_neighbor.shape[0]
                    confidence = sum(
                        1.0 for c in self.calibrationSet if c >= nonconformity) / len(self.calibrationSet)

            print("\nPredict {} with confidence {}\n".format(label, confidence))

            neighbors = []
            for i in range(len(neighborPeopleIds)):
                neighbors.append({
                    'peopleId': neighborPeopleIds[i],
                    'distance': neighborDistances[i]
                })
            msg = {
                "type": "CLASSIFIED",
                "peopleId": peopleId,
                "label": label,
                "confidence": confidence,
                "neighbors": neighbors,
            }

            self.sendMessage(json.dumps(msg))

        return peopleId, label, confidence

    def processFrame(self, msg):

        self.processCount += 1
        localProcessCount = self.processCount
        benchmark.startAvg(10.0, "processFrame")
        try:
            if args.verbose:
                print("Thread pool size: {}".format(
                    len(reactor.getThreadPool().threads)))

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

            self.logProcessTime(
                "0_start", "Start processing frame {}".format(keyframe), robotId, videoId, keyframe)

            head = "data:image/jpeg;base64,"
            assert(dataURL.startswith(head))
            imgData = base64.b64decode(dataURL[len(head):])
            imgStr = StringIO.StringIO()
            imgStr.write(imgData)
            imgStr.seek(0)
            imgPIL = Image.open(imgStr)

            self.logProcessTime(
                "1_open_image", 'Open PIL Image from base64', robotId, videoId, keyframe)

            img = np.asarray(imgPIL)

            if args.saveImg:
                imgPIL.save(os.path.join(args.imgPath, 'input',
                                         '{}-{}_{}.jpg'.format(robotId, videoId, keyframe)))
                self.logProcessTime(
                    "2_save_image", 'Save input image', robotId, videoId, keyframe)

            if args.facePredictor:
                benchmark.start(
                    "cnn_face_detector_{}".format(localProcessCount))
                bbs = cnn_face_detector(img, 0)
                benchmark.update(
                    "cnn_face_detector_{}".format(localProcessCount))
                tag, elasped, rate = benchmark.end(
                    "cnn_face_detector_{}".format(localProcessCount))
                if rate:
                    if len(bbs) > 0:
                        benchmark.logInfo("{} facePredictor_found : {:.2f}, {:.4f}, {}, {}".format(
                            tag, rate, elasped, img.shape, len(bbs)))
                    else:
                        benchmark.logInfo("{} facePredictor_not_found : {:.2f}, {:.4f}, {}, {}".format(
                            tag, rate, elasped, img.shape, len(bbs)))

            else:
                benchmark.start("hog_detector_{}".format(localProcessCount))
                bbs = hog_detector(img, 1)
                benchmark.update("hog_detector_{}".format(localProcessCount))
                benchmark.end("hog_detector_{}".format(localProcessCount))

            print("Number of faces detected: {}".format(len(bbs)))
            self.logProcessTime(
                "3_face_detected", 'Detector get face bounding box', robotId, videoId, keyframe)
            benchmark.update("processFrame")

            for index, bb in enumerate(bbs):
                if args.facePredictor:
                    print("Face detection confidence = {}".format(bb.confidence))
                    if bb.confidence < 0.8:
                        print("Drop low confidence face detection")
                        continue
                    bb = bb.rect

                print("bb width = {}, height = {}".format(
                    bb.width(), bb.height()))

                if ((bb.left() < 0) | (bb.right() < 0) | (bb.top() < 0) | (bb.bottom() < 0)):
                    continue

                # Create base64 cropped image
                cropImg = img[bb.top():bb.bottom(),
                              bb.left():bb.right()]
                _, jpgImg = cv2.imencode(
                    '.jpg', cv2.cvtColor(cropImg, cv2.COLOR_RGB2BGR))
                content = base64.b64encode(jpgImg)
                self.logProcessTime(
                    "4_crop_image", 'Crop image', robotId, videoId, keyframe)

                phash = str(imagehash.phash(Image.fromarray(cropImg)))

                grayImg = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                cropGrayImg = cv2.cvtColor(cropImg, cv2.COLOR_RGB2GRAY)

                laplacianImg = cv2.Laplacian(cropGrayImg, cv2.CV_64F)
                focus_measure = laplacianImg.var()

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

                if bb.width() < args.minFaceResolution or bb.height() < args.minFaceResolution:
                    foundFace = Face(None, None, phash=phash, content=content)
                    self.foundUser(robotId, videoId, keyframe, foundFace)
                    continue

                headPose = hpp(grayImg, bb)
                headPoseImage, p1, p2 = hp.pose_estimate(grayImg, headPose)
                headPoseLength = cv2.norm(
                    np.array(p1) - np.array(p2)) / bb.width() * 100
                print("Head Pose Length: {}".format(headPoseLength))

                cropGrayImg = headPoseImage[bb.top():bb.bottom(),
                                            bb.left():bb.right()]
                sideFace = headPoseLength > args.sideFaceThreshold

                if args.maxThreadPoolSize == 1:
                    eyes = eye_cascade.detectMultiScale(cropGrayImg)
                    for (ex, ey, ew, eh) in eyes:
                        cv2.rectangle(cropGrayImg, (ex, ey),
                                      (ex+ew, ey+eh), (0, 255, 0), 2)
                    cv2.putText(cropGrayImg, 'Side' if sideFace else 'Front',
                                (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))
                    cv2.imshow('Head Pose', cropGrayImg)
                    cv2.waitKey(1)
                    if args.saveImg:
                        cv2.imwrite(
                            "images/side_"+datetime.now().strftime("%Y-%m-%d_%H:%M:%S.%f")+".jpg", cropGrayImg)

                if sideFace:
                    print("Drop non-frontal face")
                    foundFace = Face(None, None, phash=phash, content=content)
                    self.foundUser(robotId, videoId, keyframe, foundFace)
                    continue

                if args.faceRecognitionModel:
                    benchmark.start("compute_face_descriptor_{}_{}".format(
                        localProcessCount, index))
                    shape = sp(img, bb)
                    rep = np.array(
                        fr_model.compute_face_descriptor(img, shape))
                    benchmark.update("compute_face_descriptor_{}_{}".format(
                        localProcessCount, index))
                    benchmark.end("compute_face_descriptor_{}_{}".format(
                        localProcessCount, index))
                else:
                    continue

                self.logProcessTime(
                    "6_feed_network", 'Neural network forward pass', robotId, videoId, keyframe)

                recentFaceId = self.getRecentFace(rep)
                if recentFaceId is None or self.processRecentFace:
                    faceId = recentFaceId if recentFaceId is not None else self.faceId
                    if self.enableClassifier:
                        peopleId, label, confidence = self.classifyFace(rep)

                        self.logProcessTime(
                            "7_predict_face", 'Face Prediction', robotId, videoId, keyframe)

                        if confidence and confidence > args.confidenceThreshold:
                            if peopleId in self.recentPeople:
                                timeDiff = datetime.now() - \
                                    self.recentPeople[peopleId]['time']

                                if timeDiff.total_seconds() > args.recentFaceTimeout:
                                    del self.recentPeople[peopleId]
                                else:
                                    faceId = self.recentPeople[peopleId]['faceId']

                            self.recentPeople[peopleId] = {
                                'faceId': faceId,
                                'time': datetime.now()
                            }

                            foundFace = Face(
                                rep, peopleId, faceId, phash, content, label)
                        else:
                            print("Drop unconfident face classification")
                            foundFace = self.createUnknownFace(
                                rep, faceId, phash, content)
                    else:
                        if msg.has_key("label"):
                            label = msg['label']
                            print("Found face with label: {}".format(label))
                        else:
                            label = None
                        foundFace = Face(rep, None, faceId,
                                         phash, content, label)

                    self.foundUser(robotId, videoId, keyframe, foundFace)
                else:
                    continue
            benchmark.updateAvg("processFrame")
            print("Finished processing frame {} for {} seconds.".format(
                keyframe, time.time() - start))
        except:
            print(traceback.format_exc())

    def logProcessTime(self, step, logMessage, robotId, videoId, keyframe):
        if args.verbose or benchmark.enable:
            currentTime = time.time()
            print("Keyframe: {} Step: {} for {} seconds. >> {}".format(
                keyframe, step, currentTime - self.lastLogTime, logMessage))
            self.lastLogTime = currentTime

            msg = {
                "type": "LOG",
                "robotId": robotId,
                "videoId": videoId,
                "keyframe": keyframe,
                "step": step,
                "time": datetime.now().isoformat()
            }
            self.sendMessage(json.dumps(msg))


def main(reactor):
    observer = log.startLogging(sys.stdout)
    observer.timeFormat = "%Y-%m-%d %T.%f"
    factory = WebSocketServerFactory()
    # factory.setProtocolOptions(utf8validateIncoming=False)
    factory.protocol = OpenFaceServerProtocol
    # ctx_factory = DefaultOpenSSLContextFactory(tls_key, tls_crt)
    # reactor.listenSSL(args.port, factory, ctx_factory)
    reactor.listenTCP(args.port, factory)
    reactor.run()
    return Deferred()


if __name__ == '__main__':
    task.react(main)

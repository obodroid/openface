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
import config

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

import facenet
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

args = config.loadConfig()
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
            nullMsg = {"type": "NULL"}
            self.pushMessage(nullMsg)
        elif msg['type'] == "FRAME":
            benchmark.startAvg(10.0, "processFrame")
            print("\n on message: FRAME \n")

            from datetime import datetime
            from time import sleep

            now = datetime.now()
            time_diff = now - \
                datetime.strptime(msg['time'], '%Y-%m-%dT%H:%M:%S.%fZ')
            print("frame latency: {}".format(time_diff))

            if time_diff.seconds < 1:
                facenet.putLoad(msg, self.foundFaceCallback)
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
            self.pushMessage(newMsg)
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
        elif msg['type'] == "ECHO":
            print("ECHO - {}".format(msg['time']))
            self.pushMessage(msg)
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
                                  identity=jsImage['identity'])

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
                tsne = TSNE(n_components=2, n_iter=10000,
                            random_state=0, perplexity=p)
                X_r = tsne.fit_transform(X_pca)

                print("Label encoder inverse transform")
                label_ids = self.le.inverse_transform(y)

                def getDataPoint(labelId, value, phash): return {
                    'labelId': labelId,
                    'value': value,
                    'phash': phash
                }

                self.tsneData = map(getDataPoint, label_ids,
                                    X_r.tolist(), self.images.keys())

                print("Saving working tsne to '{}'".format(self.tsneFile))
                joblib.dump(self.tsneData, self.tsneFile)

        msg = {
            "type": "TSNE_DATA",
            "data": self.tsneData
        }

        self.pushMessage(msg)

    def trainClassifier(self):
        self.tsneData = None
        d = self.getData()
        if d is None:
            self.classifier = None
            return
        else:
            (X, y) = d
            self.calibrationSet = []

            _, y_indices = np.unique(y, return_inverse=True)
            class_counts = np.bincount(y_indices)
            cond = [True if class_counts[a] > 1 else False for a in y_indices]

            X_cv = np.extract(
                np.tile(cond, (X.shape[1], 1)).transpose(), X).reshape(-1, X.shape[1])
            y_cv = np.extract(cond, y)

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
                    outlier_label=-1), param_grid=param_grid, cv=cv, n_jobs=4).fit(X_cv, y_cv)

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
        self.pushMessage(msg)

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

    def getRecentFace(self, rep, search):
        recentFaceId = None

        def isValid(face):
            timeDiff = datetime.now() - face['time']
            return timeDiff.total_seconds() < args.recentFaceTimeout

        if search:
            self.recentFaces = [] 
        else:
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
            "bbox": face.bbox,
            "time": datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'),
        }

        if face.cluster:
            msg["predict_face_id"] = face.cluster

        if face.identity:
            msg["predict_name"] = face.label
            msg["predict_people_id"] = face.identity
        elif face.label:
            msg["label"] = face.label

        self.pushMessage(msg)

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

            self.pushMessage(msg)

        return peopleId, label, confidence

    def foundFaceCallback(self, robotId, videoId, keyframe, foundFace, search = False):
        print("foundFaceCallback : {}".format(keyframe))

        if foundFace.rep is None:
            # found face but cannot recognize user
            self.foundUser(robotId, videoId, keyframe, foundFace)
            return

        # check recent face (if has face rep)
        recentFaceId = self.getRecentFace(foundFace.rep,search)
        if recentFaceId is None or self.processRecentFace:
            if recentFaceId is not None:
                faceId = recentFaceId
            else:
                faceId = self.faceId
                self.faceId += 1

            if self.enableClassifier:
                peopleId, label, confidence = self.classifyFace(foundFace.rep)

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

                    foundFace.identity = peopleId
                    foundFace.label = label
                else:
                    print("Unconfident face classification")

            foundFace.cluster = faceId
            self.foundUser(robotId, videoId, keyframe, foundFace)

        benchmark.updateAvg("processFrame")
        self.logProcessTime(
            "8_finish_process", 'Finish Processing Face', robotId, videoId, keyframe)

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
            self.pushMessage(msg)

    def pushMessage(self, msg):
        reactor.callFromThread(self.sendMessage, json.dumps(msg), sync=True)


def main(reactor):
    observer = log.startLogging(sys.stdout)
    observer.timeFormat = "%Y-%m-%d %T.%f"
    factory = WebSocketServerFactory()
    factory.setProtocolOptions(autoPingInterval=1, autoPingTimeout=2)
    factory.protocol = OpenFaceServerProtocol
    # ctx_factory = DefaultOpenSSLContextFactory(tls_key, tls_crt)
    # reactor.listenSSL(args.port, factory, ctx_factory)
    reactor.listenTCP(args.WEBSOCKET_PORT, factory)
    reactor.run()
    return Deferred()


if __name__ == '__main__':
    task.react(main)

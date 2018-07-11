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

import os
import sys
fileDir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(fileDir, "..", ".."))

import pickle
import pymongo
import pprint
import txaio
txaio.use_twisted()

from autobahn.twisted.websocket import WebSocketServerProtocol, \
    WebSocketServerFactory
from twisted.internet import task, defer
from twisted.internet.ssl import DefaultOpenSSLContextFactory

from twisted.python import log

from face import Face

import httplib, urllib
import argparse
import cv2
import imagehash
import json
from PIL import Image
import numpy as np
import StringIO
import base64
import time
import ssl
import scipy.misc

from sklearn.decomposition import PCA
from sklearn.grid_search import GridSearchCV
from sklearn.manifold import TSNE
from sklearn.svm import SVC
from sklearn.ensemble import IsolationForest
from sklearn.externals import joblib

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import openface

modelDir = os.path.join(fileDir, '..', '..', 'models')
dlibModelDir = os.path.join(modelDir, 'dlib')
openfaceModelDir = os.path.join(modelDir, 'openface')

# For TLS connections
tls_crt = os.path.join(fileDir, 'tls', 'server.crt')
tls_key = os.path.join(fileDir, 'tls', 'server.key')

parser = argparse.ArgumentParser()
parser.add_argument('--dlibFacePredictor', type=str, help="Path to dlib's face predictor.",
                    default=os.path.join(dlibModelDir, "shape_predictor_68_face_landmarks.dat"))
parser.add_argument('--networkModel', type=str, help="Path to Torch network model.",
                    default=os.path.join(openfaceModelDir, 'nn4.small2.v1.t7'))
parser.add_argument('--imgDim', type=int,
                    help="Default image dimension.", default=96)
parser.add_argument('--cuda', action='store_true')
parser.add_argument('--unknown', type=bool, default=False,
                    help='Try to predict unknown people')
parser.add_argument('--port', type=int, default=9000,
                    help='WebSocket Port')
parser.add_argument('--apiURL', type=str,
                    help="Face Server API url.", default="203.150.95.168:8540")
parser.add_argument('--workingMode', type=str,
                    help="Working mode - db_master, on_server", default="on_server")
parser.add_argument('--dth', type=str,
                    help="Representation distance threshold", default=0.5)

args = parser.parse_args()

align = openface.AlignDlib(args.dlibFacePredictor)
net = openface.TorchNeuralNet(args.networkModel, imgDim=args.imgDim,
                              cuda=args.cuda)

slideId = 0
slideWindowSize = 5
slideWindowFaces = {}

class OpenFaceServerProtocol(WebSocketServerProtocol):
    def __init__(self):
        super(OpenFaceServerProtocol, self).__init__()
        self.images = {}
        self.training = True

        if args.workingMode == "on_server":
            self.people = self.getPreTrainedModel("working_people.json",{})
            self.preSvm = self.getPreTrainedModel("working_svm.pkl")
        else:
            self.people = self.getPreTrainedModel("db_people.json",{})
            self.preSvm = self.getPreTrainedModel("db_svm.pkl")

        self.svm = None
        self.countUnknown = 0
        # self.firstPeopleRep = None if len(self.people) < 1 else self.people[0].rep
        self.unknowns = {}
        print("apiURL = "+args.apiURL)

        if args.unknown:
            self.unknownImgs = np.load("./examples/web/unknown.npy")

    def onConnect(self, request):
        print("Client connecting: {0}".format(request.peer))
        self.training = True

    def onOpen(self):
        print("WebSocket connection open.")

    def onMessage(self, payload, isBinary):
        raw = payload.decode('utf8')
        msg = json.loads(raw)
        print("Received {} message of length {}.".format(
            msg['type'], len(raw)))
        if msg['type'] == "ALL_STATE":
            self.loadState(msg['images'], msg['training'], msg['people'])
        elif msg['type'] == "NULL":
            self.sendMessage('{"type": "NULL"}')
        elif msg['type'] == "FRAME":
            self.processFrame(msg)
            self.sendMessage('{"type": "PROCESSED"}')
        elif msg['type'] == "TRAINING":
            self.training = msg['val']
            if not self.training:
                self.trainSVM()
        elif msg['type'] == "UPDATE_IDENTITY":
            h = msg['hash'].encode('ascii', 'ignore')
            if h in self.images:
                self.images[h].identity = msg['idx']
                if not self.training:
                    self.trainSVM()
            else:
                print("Image not found.")
        elif msg['type'] == "REMOVE_IMAGE":
            h = msg['hash'].encode('ascii', 'ignore')
            if h in self.images:
                del self.images[h]
                if not self.training:
                    self.trainSVM()
            else:
                print("Image not found.")
        elif msg['type'] == 'REQ_TSNE':
            self.sendTSNE(msg['people'])
        elif msg['type'] == "ECHO":
            newMsg = {
                "type": "ResponseECHO",
                "message": msg
            }
            self.sendMessage(json.dumps(newMsg))
        else:
            print("Warning: Unknown message type: {}".format(msg['type']))

    def onClose(self, wasClean, code, reason):
        print("WebSocket connection closed: {0}".format(reason))

    def loadState(self, jsImages, training, jsPeople):
        self.training = training

        for jsImage in jsImages:
            h = jsImage['hash'].encode('ascii', 'ignore')
            self.images[h] = Face(np.array(jsImage['representation']),
                                  jsImage['identity'])

        if not training:
            self.trainSVM()

    def getData(self):
        X = []
        y = []

        if len(self.images) < 5:
            return None

        for img in self.images.values():
            X.append(img.rep)
            y.append(img.identity)

        numIdentities = len(set(y + [-1])) - 1
        if numIdentities == 0:
            return None

        if args.unknown:
            numUnknown = y.count(-1)
            numIdentified = len(y) - numUnknown
            numUnknownAdd = (numIdentified / numIdentities) - numUnknown
            if numUnknownAdd > 0:
                print("+ Augmenting with {} unknown images.".format(numUnknownAdd))
                for rep in self.unknownImgs[:numUnknownAdd]:
                    X.append(rep)
                    y.append(-1)

        X = np.vstack(X)
        y = np.array(y)
        return (X, y)

    def sendTSNE(self, people):
        d = self.getData()
        if d is None:
            return
        else:
            (X, y) = d

        X_pca = PCA(n_components=50).fit_transform(X, X)
        tsne = TSNE(n_components=2, init='random', random_state=0)
        X_r = tsne.fit_transform(X_pca)

        yVals = list(np.unique(y))
        colors = cm.rainbow(np.linspace(0, 1, len(yVals)))

        plt.figure()
        for c, i in zip(colors, yVals):
            name = "Unknown" if i == -1 else people[i]
            plt.scatter(X_r[y == i, 0], X_r[y == i, 1], c=c, label=name)
            plt.legend()

        imgdata = StringIO.StringIO()
        plt.savefig(imgdata, format='png')
        imgdata.seek(0)

        content = 'data:image/png;base64,' + \
                  urllib.quote(base64.b64encode(imgdata.buf))
        msg = {
            "type": "TSNE_DATA",
            "content": content
        }
        self.sendMessage(json.dumps(msg))

    def getPreTrainedModel(self,filename,defaultValue=None):
        if os.path.isfile(filename):
            return joblib.load(filename)
        else:
            return defaultValue

    def trainSVM(self):
        print("+ Training SVM on {} labeled images.".format(len(self.images)))
        d = self.getData()
        if d is None:
            self.svm = None
            return
        else:
            (X, y) = d
            numIdentities = len(set(y + [-1]))
            print("numIdentities = {}".format(numIdentities))

            if numIdentities <= 1:
                return

            param_grid = [
                {'C': [1, 10, 100, 1000],
                 'kernel': ['linear']},
                {'C': [1, 10, 100, 1000],
                 'gamma': [0.001, 0.0001],
                 'kernel': ['rbf']}
            ]
            self.svm = GridSearchCV(SVC(C=1,probability=True,decision_function_shape='ovr'), param_grid, cv=5).fit(X, y)
            # self.isoForest = IsolationForest(max_samples=100)
            # self.isoForest.fit(X,y)
            # savePickle = pickle.dump(self.svm, open( "face_svm.pkl", "wb"))

    def hasFoundSimilarFace(self,rep):
        foundSimilarRep = False
        for previousFaces in slideWindowFaces.values():
            for previousFace in previousFaces:
                d = rep - previousFace.rep
                drep = np.dot(d, d)
                print("Squared l2 distance between representations: {:0.3f}".format(drep))
                
                if drep < args.dth:
                    print("assign previous id")
                    foundSimilarRep = True
                    # unknownIdentity = previousFace.identity
                    break
            else:
                continue  # executed if the loop ended normally (no break)
            break  # executed if 'continue' was skipped (break)

        return foundSimilarRep
    
    def newFaceIdentity(self,rep,phash=None,content=None):
        identity = len(self.people)
        # if len(self.people)==0:
        #     self.firstPeopleRep = rep

        name = "User "+str(identity)

        newFace = Face(rep, identity,phash,content,name)
        self.people[identity] = newFace
        self.images[phash] = newFace
        msg = {
            "type": "NEW_IMAGE",
            "hash": phash,
            "content": content,
            "identity": identity,
            "representation": rep.tolist()
        }
        self.sendMessage(json.dumps(msg))
        return newFace

    def foundUser(self,robotId,videoId,face):
        aiMsg = {
            "type": "FOUND_USER",
            "robotId":robotId,
            "videoId":videoId,
            "phash": face.phash,
            "content": face.content,
            "predict_face_id": face.identity,
            "rep": face.rep.tolist(),
            "time": time.time(),
            "name": face.name
        }
        self.sendMessage(json.dumps(aiMsg))

    def sendToAPI(self,msg):
        url = args.apiURL
        params = json.dumps(msg).encode('utf8')
        headers = {"Content-type": "application/json"}
        conn = httplib.HTTPSConnection(url,timeout=5, context=ssl._create_unverified_context())
        conn.request("POST", "/face_images", params,headers)
        response = conn.getresponse()
        print response.status, response.reason
        conn.close()

    def getFacesFromAPI(self):
        url = args.apiURL
        conn = httplib.HTTPSConnection(url,timeout=5, context=ssl._create_unverified_context())
        conn.request("GET", "/face_images?limit=10")
        response = conn.getresponse()
        print response.status, response.reason
        data = response.read()

        print(data)
        results = json.loads(data)
        dbFaces = results["results"]
        conn.close()
        print("results -{}".format(results))

    def getPeopleFromAPI(self,msg):
        url = args.apiURL
        params = json.dumps(msg).encode('utf8')
        conn = httplib.HTTPSConnection(url,timeout=5, context=ssl._create_unverified_context())
        conn.request("GET", "/people")
        response = conn.getresponse()
        print response.status, response.reason
        conn.close()

    def processFrame(self, msg):
        dataURL= msg['dataURL']
        identity = msg['identity']
        if msg.has_key("robotId"):
            robotId = msg['robotId'] 
        else:
            robotId = ""
        
        if msg.has_key("videoId"):
            videoId = msg['videoId'] 
        else:
            videoId = ""

        head = "data:image/jpeg;base64,"
        assert(dataURL.startswith(head))
        imgdata = base64.b64decode(dataURL[len(head):])
        imgF = StringIO.StringIO()
        imgF.write(imgdata)
        imgF.seek(0)
        img = Image.open(imgF)

        buf = np.fliplr(np.asarray(img))

        rgbFrame = np.zeros((img.height, img.width, 3), dtype=np.uint8)
        rgbFrame[:, :, 0] = buf[:, :, 2]
        rgbFrame[:, :, 1] = buf[:, :, 1]
        rgbFrame[:, :, 2] = buf[:, :, 0]

        annotatedFrame = np.copy(buf)

        global slideId
        slideWindowFaces[slideId] = []
        identities = []
        bb = align.getLargestFaceBoundingBox(rgbFrame)
        bbs = [bb] if bb is not None else []

        for bb in bbs:
            print("bb = {}".format(bb))
            print("bb width = {}, height = {}".format(bb.width(),bb.height()))

            cropImage = rgbFrame[bb.top():bb.bottom(), bb.left():bb.right()]
            print("crop image : {}".format(len(cropImage)))
            if (len(cropImage) > 0) & (bb.left() > 0) & (bb.right() > 0) & (bb.top() > 0) & (bb.bottom() > 0) :
                cv2.imshow("cropped", cropImage)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    return

                landmarks = align.findLandmarks(rgbFrame, bb)
                alignedFace = align.align(args.imgDim, rgbFrame, bb,
                                        landmarks=landmarks,
                                        landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
                if alignedFace is None:
                    continue
                
                phash = str(imagehash.phash(Image.fromarray(alignedFace)))
                print("phash = " + phash)
                
                identity = -1 #unknown
                # unknownIdentity = None
                rep = net.forward(alignedFace)
                cropImage = cropImage[:, :, ::-1].copy() # RGB to BGR for PIL image
                cropPIL = scipy.misc.toimage(cropImage)
                buf_crop = StringIO.StringIO()
                cropPIL.save(buf_crop, format="PNG")
                content = base64.b64encode(buf_crop.getvalue())
                content= 'data:image/png;base64,' + content

                # if phash in self.images:
                #     identity = self.images[phash].identity
                # else:

                if not self.hasFoundSimilarFace(rep):
                    # TODO: using new model prediction
                    if self.preSvm: # no preSvm, just face-detection (not recognition)
                        predictIdentity = self.preSvm.predict([rep])[0]
                        name = self.people[predictIdentity]
                        foundFace = Face(rep, predictIdentity, phash, content, name)

                        # TODO: need condition to check confidence or probability that is much enough to decide if person is unknown or not

                        # might not necessary to recheck again
                        # predictFace = self.people[predictIdentity]
                        # d = rep - predictFace.rep
                        # drep = np.dot(d, d)
                        # print("distance of prediction: {:0.3f}".format(drep))
                        # if drep > args.dth:
                        #     foundFace = self.newFaceIdentity(rep,phash,content)
                        # else :
                        #     foundFace = Face(rep, predictIdentity,phash,content,name)

                        # if msg.has_key("sourceFolder"):
                        #     print("identity - {}, name - {}, drep - {}, imageFile - {}".format(identity,predictPeople,drep,msg["imageFile"]))
                        #     return
                    else :
                        foundFace = self.newFaceIdentity(rep,phash,content)
                    
                    slideWindowFaces[slideId].append(foundFace)
                    identity = foundFace.identity
                    self.foundUser(robotId,videoId,foundFace)
                else :
                    # just drop similar face, do nothing
                    continue

                if identity not in identities:
                    identities.append(identity)

                msg = {
                    "type": "IDENTITIES",
                    "identities": identities
                }
                self.sendMessage(json.dumps(msg))

                plt.figure()
                plt.imshow(annotatedFrame)
                plt.xticks([])
                plt.yticks([])

                imgdata = StringIO.StringIO()
                plt.savefig(imgdata, format='png')
                imgdata.seek(0)
                content = 'data:image/png;base64,' + \
                    urllib.quote(base64.b64encode(imgdata.buf))
                msg = {
                    "type": "ANNOTATED",
                    "content": content
                }
                plt.close()

        slideId +=1

def main(reactor):
    log.startLogging(sys.stdout)
    factory = WebSocketServerFactory()
    factory.protocol = OpenFaceServerProtocol
    ctx_factory = DefaultOpenSSLContextFactory(tls_key, tls_crt)
    reactor.listenSSL(args.port, factory, ctx_factory)
    return defer.Deferred()

if __name__ == '__main__':
    task.react(main)

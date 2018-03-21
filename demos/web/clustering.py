#!/usr/bin/env python2
#
# Copyright 2018 Obodroid Corporation by Lertlove
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
import shutil
import pprint
fileDir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(fileDir, "..", ".."))

# import pickle
# import pymongo
import pymongo
from pymongo import MongoClient

# import pickle
from sklearn.externals import joblib

# from autobahn.twisted.websocket import WebSocketServerProtocol, \
#     WebSocketServerFactory
# from twisted.internet import task, defer
# from twisted.internet.ssl import DefaultOpenSSLContextFactory

# from twisted.python import log

# import httplib, urllib
import argparse
# import cv2
import imagehash
# import json
from PIL import Image
import numpy as np
import StringIO
import base64
# import time
# import ssl

# from sklearn.decomposition import PCA
from sklearn.grid_search import GridSearchCV
# from sklearn.manifold import TSNE
from sklearn.svm import SVC
from sklearn.cluster import DBSCAN

# import matplotlib as mpl
# mpl.use('Agg')
# import matplotlib.pyplot as plt
# import matplotlib.cm as cm

import openface

modelDir = os.path.join(fileDir, '..', '..', 'models')
dlibModelDir = os.path.join(modelDir, 'dlib')
openfaceModelDir = os.path.join(modelDir, 'openface')
# # For TLS connections
# tls_crt = os.path.join(fileDir, 'tls', 'server.crt')
# tls_key = os.path.join(fileDir, 'tls', 'server.key')

parser = argparse.ArgumentParser()
parser.add_argument('--dlibFacePredictor', type=str, help="Path to dlib's face predictor.",
                    default=os.path.join(dlibModelDir, "shape_predictor_68_face_landmarks.dat"))
parser.add_argument('--networkModel', type=str, help="Path to Torch network model.",
                    default=os.path.join(openfaceModelDir, 'nn4.small2.v1.t7'))
parser.add_argument('--imgDim', type=int,
                    help="Default image dimension.", default=96)
parser.add_argument('--cuda', action='store_true')
# parser.add_argument('--unknown', type=bool, default=False,
#                     help='Try to predict unknown people')
# parser.add_argument('--port', type=int, default=9000,
#                     help='WebSocket Port')
# parser.add_argument('--apiURL', type=str,
#                     help="Face Server API url.", default="192.168.1.243:8540")
# parser.add_argument('--mongoURL', type=str,
#                     help="Mongo DB url.", default="203.150.95.168:8540")

parser.add_argument('--mongoURL', type=str,
                    help="Mongo DB url.", default="192.168.1.243:27017")
parser.add_argument('--sourceFolder', type=str,
                    help="Source Folder", default="./tests/captured_images")
parser.add_argument('--targetFolder', type=str,
                    help="Target Folder", default="./tests/cluster_images")

args = parser.parse_args()

align = openface.AlignDlib(args.dlibFacePredictor)
net = openface.TorchNeuralNet(args.networkModel, imgDim=args.imgDim,
                              cuda=args.cuda)

class ClusteringServer:
    def __init__(self):
        # self.mongoURL = args.mongoURL
        # self.client = MongoClient("mongodb://"+self.mongoURL)
        # self.db = self.client.robot
        self.sourceFolder = args.sourceFolder
        self.targetFolder = args.targetFolder


    def prepareData(self,path):
        self.X = []
        self.Y = []
        for filename in os.listdir(path):
            if not filename.endswith('.jpg'):
                continue
            # image = open(path+"/"+filename, 'rb') #open binary file in read mode
            # image_read = image.read()
            # imgdata = base64.encodestring(image_read)
            filepath = os.path.join(path, filename)
            img = Image.open(filepath)
            # head = "data:image/jpeg;base64,"
            # imgdata = base64.b64decode(dataURL[len(head):])
            # print(imgdata)

            # imgF = StringIO.StringIO()
            # imgF.write(imgdata)
            # imgF.seek(0)
            # img = Image.open(imgF)

            buf = np.fliplr(np.asarray(img))
            rgbFrame = np.zeros((300, 400, 3), dtype=np.uint8)
            rgbFrame[:, :, 0] = buf[:, :, 2]
            rgbFrame[:, :, 1] = buf[:, :, 1]
            rgbFrame[:, :, 2] = buf[:, :, 0]

            bb = align.getLargestFaceBoundingBox(rgbFrame)
            bbs = [bb] if bb is not None else []

            for bb in bbs:
                # print(len(bbs))
                landmarks = align.findLandmarks(rgbFrame, bb)
                alignedFace = align.align(args.imgDim, rgbFrame, bb,
                                          landmarks=landmarks,
                                          landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
                if alignedFace is None:
                    continue
                
                phash = str(imagehash.phash(Image.fromarray(alignedFace)))
                print("phash = "+phash)
                
                rep = net.forward(alignedFace)
                self.X.append(rep)
                self.Y.append(filename)
        
    def cluster(self):
        db = DBSCAN(eps=0.3, min_samples=2).fit(self.X)
        labels = db.labels_
        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        print('Estimated number of clusters: %d' % n_clusters_)
        for index, label in enumerate(labels):
            filename = self.Y[index]
            label = "User_" + str(label)
            print("index - {}, label - {}, filename - {}".format(index,label,filename))

            source = os.path.join(self.sourceFolder, filename)
            # if not os.path.exists(self.targetFolder):
            #     os.makedirs(self.targetFolder)

            destination = os.path.join(self.targetFolder, label)
            if not os.path.exists(destination):
                os.makedirs(destination)
            destination = os.path.join(destination, filename)
            shutil.copyfile(source, destination)

        return None


def main(reactor):
    clusteringServer = ClusteringServer()
    print("Clustering people in folder : "+clusteringServer.sourceFolder)
    clusteringServer.prepareData(clusteringServer.sourceFolder)
    clusteringServer.cluster()

    

if __name__ == '__main__':
    main(sys.argv)

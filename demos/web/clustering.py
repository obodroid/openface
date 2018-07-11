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

from PIL import Image
import cv2
import pymongo
from pymongo import MongoClient

from sklearn.externals import joblib

import argparse
import imagehash
from PIL import Image
import numpy as np
import StringIO
import base64
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
from sklearn.cluster import DBSCAN

import openface

modelDir = os.path.join(fileDir, '..', '..', 'models')
dlibModelDir = os.path.join(modelDir, 'dlib')
openfaceModelDir = os.path.join(modelDir, 'openface')

parser = argparse.ArgumentParser()
parser.add_argument('--dlibFacePredictor', type=str, help="Path to dlib's face predictor.",
                    default=os.path.join(dlibModelDir, "shape_predictor_68_face_landmarks.dat"))
parser.add_argument('--networkModel', type=str, help="Path to Torch network model.",
                    default=os.path.join(openfaceModelDir, 'nn4.small2.v1.t7'))
parser.add_argument('--imgDim', type=int,
                    help="Default image dimension.", default=96)
parser.add_argument('--cuda', action='store_true')

parser.add_argument('--mongoURL', type=str,
                    help="Mongo DB url.", default="192.168.1.243:27017")
parser.add_argument('--sourceFolder', type=str,
                    help="Source Folder", default="./tests/captured_images")
parser.add_argument('--targetFolder', type=str,
                    help="Target Folder", default="./tests/cluster_images")
parser.add_argument('--mode', type=str,
                    help="Function Mode", default="detect")
parser.add_argument('--dth', type=float,
                    help="Representation distance threshold", default=0.5)

args = parser.parse_args()

align = openface.AlignDlib(args.dlibFacePredictor)
net = openface.TorchNeuralNet(args.networkModel, imgDim=args.imgDim,
                              cuda=args.cuda)

class ClusteringServer:
    def __init__(self):
        self.sourceFolder = args.sourceFolder
        self.targetFolder = args.targetFolder

    def prepareData(self,path):
        self.X = []
        self.Y = []
        for filename in os.listdir(path):
            if not filename.endswith('.jpg'):
                continue
            filepath = os.path.join(path, filename)
            try:
                img = Image.open(filepath)
            except:
                print("cannot open image file")
                continue

            baseFileName = os.path.splitext(os.path.basename(filename))[0]
            rgbFrame = self.convertImageToRgbFrame(img)

            bbs = align.getAllFaceBoundingBoxes(rgbFrame)

            faceInFile=0

            for bb in bbs:
                faceInFile+=1
                cropImage = rgbFrame[bb.top():bb.bottom(), bb.left():bb.right()]
                print("crop image : {}".format(len(cropImage)))
                if (len(cropImage) > 0) & (bb.left() > 0) & (bb.right() > 0) & (bb.top() > 0) & (bb.bottom() > 0) :
                    cv2.imshow("cropped", cropImage)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        return

                    cropFolder = os.path.join(self.targetFolder, "crop")
                    if not os.path.exists(cropFolder):
                        os.makedirs(cropFolder)
                    
                    cropFile = baseFileName+"-"+str(faceInFile)+".jpg"
                    cropPath = os.path.join(cropFolder, cropFile)

                    im = Image.fromarray(cropImage)
                    im.save(cropPath)

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
                    self.Y.append(cropFile)
        
    def cluster(self):
        db = DBSCAN(eps=0.5, min_samples=2).fit(self.X)
        labels = db.labels_
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        print('Estimated number of clusters: %d' % n_clusters_)
        for index, label in enumerate(labels):
            filename = self.Y[index]
            label = "User_" + str(label)
            print("index - {}, label - {}, filename - {}".format(index,label,filename))

            source = os.path.join(self.targetFolder, "crop")
            source = os.path.join(source, filename)

            destination = os.path.join(self.targetFolder, label)
            if not os.path.exists(destination):
                os.makedirs(destination)
            destination = os.path.join(destination, filename)
            shutil.copyfile(source, destination)

        return None
    
    def detect(self):
        base = os.path.join(self.sourceFolder, "base")
        foundCnt = 0
        for filename in os.listdir(base):
            if not filename.endswith('.jpg'):
                continue
            print("base filename = "+filename)
            filepath = os.path.join(base, filename)
            img = Image.open(filepath)

            rgbFrame = self.convertImageToRgbFrame(img)
            bb = align.getLargestFaceBoundingBox(rgbFrame)
            landmarks = align.findLandmarks(rgbFrame, bb)
            alignedFace = align.align(args.imgDim, rgbFrame, bb,
                                    landmarks=landmarks,
                                    landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
            if alignedFace is None:
                continue
            
            phash = str(imagehash.phash(Image.fromarray(alignedFace)))
            
            baseRep = net.forward(alignedFace)
            for index, cropFile in enumerate(self.Y):
                cropFolder = os.path.join(self.targetFolder, "crop")
                source = os.path.join(cropFolder, cropFile)
                if os.path.exists(source):
                    print("index = {}, cropFile = {}".format(index,cropFile))
                    rep = self.X[index]
                    d = baseRep - rep
                    drep = np.dot(d, d)
                    print("Squared l2 distance between representations: {:0.3f}".format(drep))
                    
                    if drep < args.dth:
                        print("found user")
                        foundCnt = foundCnt+1
                        destination = os.path.join(self.targetFolder, "found")

                        if not os.path.exists(destination):
                            os.makedirs(destination)
                        destination = os.path.join(destination, cropFile)
                        shutil.move(source, destination)
        print("foundCnt - {}".format(foundCnt))
            
    def convertImageToRgbFrame(self,img):
        imarr = np.asarray(img)
        buf = np.fliplr(np.asarray(img))
        rgbFrame = np.zeros((img.height, img.width, 3), dtype=np.uint8)
        rgbFrame[:, :, 0] = buf[:, :, 2]
        rgbFrame[:, :, 1] = buf[:, :, 1]
        rgbFrame[:, :, 2] = buf[:, :, 0]
        return rgbFrame

def main(reactor):
    clusteringServer = ClusteringServer()
    print("Clustering people in folder : "+clusteringServer.sourceFolder)
    clusteringServer.prepareData(clusteringServer.sourceFolder)
    
    if args.mode == "cluster":
        clusteringServer.cluster()
    else:
        clusteringServer.detect()

if __name__ == '__main__':
    main(sys.argv)

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

import sys
import pprint
import pymongo
from pymongo import MongoClient

from sklearn.externals import joblib
import argparse
import numpy as np
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import IsolationForest

parser = argparse.ArgumentParser()

parser.add_argument('--mongoURL', type=str,
                    help="Mongo DB url.", default="192.168.1.243:27017")

args = parser.parse_args()

class TrainingServer:
    def __init__(self):
        self.mongoURL = args.mongoURL
        self.client = MongoClient("mongodb://"+self.mongoURL)
        self.db = self.client.robot
        self.images = None

    def getFaces(self):
        return list(self.db.faces.find())

    def prepareData(self,images):
        X = []
        y = []

        if len(images) < 5:
            return None

        for img in images:
            X.append(img['rep'])
            y.append(int(img['face_id']))

        numIdentities = len(set(y + [-1])) - 1
        if numIdentities == 0:
            return None

        X = np.vstack(X)
        y = np.array(y)
        return (X, y)

    def trainFaces(self,images):
        print("+ Training SVM on {} labeled images.".format(len(self.images)))
        d = self.prepareData(images)
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
            self.isoForest = IsolationForest(max_samples=100)
            self.isoForest.fit(X,y)
            svmPickle = joblib.dump(self.svm, 'db_svm.pkl')

def main(reactor):

    trainingServer = TrainingServer()
    print("trainingServer.mongoURL - "+trainingServer.mongoURL)
    trainingServer.images = trainingServer.getFaces()
    trainingServer.trainFaces(trainingServer.images)
    
if __name__ == '__main__':
    main(sys.argv)

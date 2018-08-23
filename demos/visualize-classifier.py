#!/usr/bin/env python2
#
# Example to classify faces.
# Brandon Amos
# 2015/10/11
#
# Copyright 2015-2016 Carnegie Mellon University
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
ptvsd.enable_attach("my_secret", address=('0.0.0.0', 3000))
# Pause the program until a remote debugger is attached
ptvsd.wait_for_attach()

import time

start = time.time()

import argparse
import cv2
import os
import pickle
import sys
import shutil
import dill
import json

from operator import itemgetter

import numpy as np
np.set_printoptions(precision=2)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import openface

from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from sklearn.mixture import GMM
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.externals import joblib


def infer(args):
    with open(args.classifierModel, 'rb') as f:
        if sys.version_info[0] < 3:
            (le, clf) = pickle.load(f)
        else:
            (le, clf) = pickle.load(f, encoding='latin1')

    data = np.arange(-2, 2, 0.2)
    results = np.zeros(shape=(len(data), len(data)))

    for indexI, valueI in enumerate(data):
        for indexJ, valueJ in enumerate(data):
            try:
                reps = [(2, np.array([valueI, valueJ]))]
            except:
                continue

            if len(reps) > 1:
                print("List of faces in image from left to right")

            for r in reps:
                rep = r[1].reshape(1, -1)
                bbx = r[0]
                start = time.time()
                predictions = clf.predict_proba(rep).ravel()
                maxI = np.argmax(predictions)
                person = le.inverse_transform(maxI)
                confidence = predictions[maxI]
                name = person.decode('utf-8')
                results[indexI, indexJ] = confidence

                if args.verbose:
                    print("Prediction took {} seconds.".format(
                        time.time() - start))

                print("Predict {},{} as {} with {:.2f} confidence.".format(
                    valueI, valueJ, name, confidence))

                if isinstance(clf, GMM):
                    dist = np.linalg.norm(rep - clf.means_[maxI])
                    print("  + Distance from the mean: {}".format(dist))

    _x, _y = np.meshgrid(data, data)
    x, y = _x.ravel(), _y.ravel()
    top = results.ravel()
    bottom = np.zeros_like(top)
    width = depth = 0.2
    fig = plt.figure(figsize=(24, 9))
    axis = fig.add_subplot(121, projection='3d')
    axis.bar3d(x, y, bottom, width, depth, top, shade=True)
    outputPath = "{}.png".format(os.path.join(args.outputDir, 'confidence'))
    plt.savefig(outputPath)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose', action='store_true')

    subparsers = parser.add_subparsers(dest='mode', help="Mode")

    inferParser = subparsers.add_parser(
        'infer', help='Predict who an image contains from a trained classifier.')
    inferParser.add_argument(
        'classifierModel',
        type=str,
        help='The Python pickle representing the classifier. This is NOT the Torch network model, which can be set with --networkModel.')
    inferParser.add_argument('--outputDir')

    start = time.time()

    args = parser.parse_args()
    if args.verbose:
        print("Loading the dlib and OpenFace models took {} seconds.".format(
            time.time() - start))
        start = time.time()

    if args.mode == 'infer':
        infer(args)

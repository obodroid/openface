from imutils.video import FPS
from random import randint
import os, signal
import sys
from datetime import datetime
import time
import cv2
from threading import Timer
import logging

log = logging.getLogger() # 'root' Logger
console = logging.StreamHandler()
timeNow = datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H:%M:%S')
logFile = logging.FileHandler("/src/benchmark/benchmark_{}.log".format(timeNow))
saveDir = "/src/benchmark/images/"

format_str = '%(asctime)s\t%(levelname)s -- %(processName)s %(filename)s:%(lineno)s -- %(message)s'
console.setFormatter(logging.Formatter(format_str))
logFile.setFormatter(logging.Formatter(format_str))

log.addHandler(console) # prints to console.
log.addHandler(logFile) # prints to console.
log.setLevel(logging.DEBUG) # anything ERROR or above
# log.warn('Import darknet.py!')
# log.critical('Going to load neural network over GPU!')

mode = 'benchmark'
benchmarks = {}
imageCount = 0

def startAvg(period,tag):
    if tag not in benchmarks and mode == 'benchmark' :
        print("startAvg {}".format(tag))
        fps = FPS().start()
        benchmarks[tag] = fps
        t = Timer(period, endAvg, [tag])
        t.start()

def updateAvg(tag):
    # print("updateBenchmark {}".format(tag))
    if tag in benchmarks:
        benchmarks[tag].update()

def endAvg(tag):
    print("endAvg {}".format(tag))
    if tag in benchmarks:
        fps = benchmarks[tag]
        fps.stop()
        log.info("{} rate: {:.2f}".format(tag,fps.fps()))
        del benchmarks[tag]

def start(tag):
    if tag not in benchmarks and mode == 'benchmark' :
        # print("start {}".format(tag))
        fps = FPS().start()
        benchmarks[tag] = fps

def update(tag):
    if tag in benchmarks:
        benchmarks[tag].update()

def end(tag):
    if tag in benchmarks:
        # print("endBenchmark {}".format(tag))
        fps = benchmarks[tag]
        fps.stop()
        log.info("{} rate: {:.2f}".format(tag,fps.fps()))
        del benchmarks[tag]

def saveImage(img,label):
    global imageCount
    if mode == 'benchmark' :
        imageCount += 1
        filename = '{}'.format(imageCount)
        filepath = '{}/{}/{}.jpg'.format(saveDir,label,filename)

        if not os.path.exists(os.path.dirname(filepath)):
            try:
                os.makedirs(os.path.dirname(filepath))
            except OSError as exc: # Guard against race condition
                print "OSError:cannot make directory."
        cv2.imwrite(filepath,img)

def logInfo(msg):
    log.info(msg)
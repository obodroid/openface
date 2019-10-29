
import os
import sys

environment = "dev"

fileDir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(fileDir, ".."))

# config.py
class Config:

    HTTP_PORT = 8000
    WEBSOCKET_PORT = 9000
    NUM_WORKERS = 4
    NUM_GPUS = 1

    modelDir = os.path.join(fileDir, '..', 'models')
    dlibModelDir = os.path.join(modelDir, 'dlib')
    openfaceModelDir = os.path.join(modelDir, 'openface')

    imgPath = os.path.join(fileDir, '..', 'data')
    shapePredictor = os.path.join(dlibModelDir, "shape_predictor_5_face_landmarks.dat")
    headPosePredictor = os.path.join(dlibModelDir, "shape_predictor_68_face_landmarks.dat")
    facePredictor = os.path.join(dlibModelDir, "mmod_human_face_detector.dat") # or null to use hog
    eyeCascade = os.path.join(modelDir, "haarcascade_eye.xml")
    faceRecognitionModel = os.path.join(dlibModelDir, 'dlib_face_recognition_resnet_model_v1.dat')
    
    recentFaceTimeout=10
    minFaceResolution=100
    loosenFactor=1.0
    focusMeasure=50
    sideFaceThreshold=8
    confidenceThreshold=0.2
    dth=0.2
    classifier="RadiusNeighbors" #choices=['SVC','RadiusNeighbors']

    # # For TLS connections
    # tls_crt = os.path.join(fileDir, 'tls', 'server.crt')
    # tls_key = os.path.join(fileDir, 'tls', 'server.key')
    
    # For connect with Facepp
    keyFacepp = "eoYSb8GL-d54k3_C37K7XfHxLcLfNoug"
    secretFacepp = "x6eLXTXf1ORSXlJeh9Zpcf9t5-HCadn-"
      
class DevConfig(Config):
    DEBUG = True
    verbose = True
    saveImg = False
    
class TestConfig(Config):
    DEBUG = True
    TESTING = True
    saveImg = True

class ProdConfig(Config):
    DEBUG = False
    TESTING = False
    saveImg = False

def loadConfig(env=None):
    if env is None:
        env = environment

    if env == 'dev':
        return DevConfig
    elif env == 'test':
        return TestConfig
    elif env == 'prod':
        return ProdConfig
    else:
        raise ValueError('Invalid environment name')

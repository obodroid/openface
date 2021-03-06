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

#import for use Face++
import requests
import json
import config
args = config.loadConfig()

class Facepp():
    def __init__(self):
        #request import requests
        self.myKey = args.keyFacepp
        self.mySecret = args.secretFacepp
        self.path = "./../../data/imageType/"
        self.mapIndexHeadPoseDict = {'1': 'top-Left headpose', '2': 'top-Mid headpose', '3': 'top-Right headpose',
                                    '4': 'mid-Left headpose', '5': 'mid-Mid headpose', '6': 'mid-Right headpose',
                                    '7': 'bot-Left headpose', '8': 'bot-Mid headpose', '9': 'bot-Right headpose',
                                    '0': 'unknown headpose'}
        self.age = None
        self.gender = None
        self.ethnicity = None
        self.emotion = None
        self.mouth = None
        self.lefteyeStatus = None
        self.righteyeStatus = None
        self.facequality = None
        self.headpose = None
        self.indexFace = None
        self.comment = [2,5]


    def findMaxValueInDict(self, myDict):
   	   inverse = [(value, key) for key, value in myDict.items()]
   	   return max(inverse)[1]

    def detect(self, picBase64):
        attributes="gender,age,ethnicity,mouthstatus,eyestatus,facequality,emotion,headpose"
        
        #url for contact with Face++ detect
        http_url = 'https://api-us.faceplusplus.com/facepp/v3/detect'

        try:
            #get values from Face++ 
            jsonResp = requests.post(http_url,
                            data = { 
                                'api_key': self.myKey,
                                'api_secret': self.mySecret,
                                'image_base64': picBase64,
                                'return_attributes': attributes
                                }
                            )	  
            #transfrom json text to dictionary
            contentObj = json.loads(vars(jsonResp)['_content'])
            attributesDict = contentObj['faces'][0]['attributes']
            
            #age value
            self.age = attributesDict['age'].values()[0]
            #gender value (Male or Female)
            self.gender = attributesDict['gender'].values()[0]
            #ethnicity value (Asian ,White, Black)
            self.ethnicity = attributesDict['ethnicity'].values()[0]
            #emotion (anger ,disgust, fear, happiness, neutral, sadness, surprise)
            self.emotion = self.findMaxValueInDict(attributesDict['emotion'])
            #mouth (surgical_mask_or_respirator, other_occlusion, close, open)
            self.mouth = self.findMaxValueInDict(attributesDict['mouthstatus'])
            #eye status (occlusion, no_glass_eye_open, normal_glass_eye_close, normal_glass_eye_open, dark_glasses, no_glass_eye_close)
            self.lefteyeStatus = self.findMaxValueInDict(attributesDict['eyestatus']['left_eye_status'])
            self.righteyeStatus = self.findMaxValueInDict(attributesDict['eyestatus']['right_eye_status'])
            #if threshold is less than values, this picture can comparable.
            faceth = self.findMaxValueInDict(attributesDict['facequality'])
            if faceth == 'value':
                self.facequality = 'high'
            else:
                self.facequality = 'low'
            self.headpose = attributesDict['headpose']
        except:
            pass

    def save9typeOfFace(self, nameImage, inputImage):
		# index = 1 (top-left), index = 2 (top-mid), index = 3 (top-right)
		# index = 4 (mid-left), index = 5 (mid-mid), index = 6 (mid-right)
		# index = 7 (bottom-left), index = 8 (bottom-mid), index = 9 (bottom-right)
		cv2.imwrite(os.path.join(self.path + self.mapIndexHeadPoseDict[self.indexFace] , nameImage), inputImage)

    def found9typeOfFace(self):
       	# index = 1 (top-left), index = 2 (top-mid), index = 3 (top-right)
		# index = 4 (mid-left), index = 5 (mid-mid), index = 6 (mid-right)
		# index = 7 (bottom-left), index = 8 (bottom-mid), index = 9 (bottom-right)
        indexFace = '0'

        if self.headpose != None:
            if self.headpose['yaw_angle'] > 15:
                yaw = 1 # left
            elif self.headpose['yaw_angle'] < -15:
                yaw = 3 # right
            else:
                yaw = 2 # mid

            if self.headpose['pitch_angle'] < -15:
                pitch = 1 # top
            elif self.headpose['pitch_angle'] > 15:
                pitch = 3 # bottom
            else:
                pitch = 2 # mid

            indexFace = str(3 * (pitch - 1) + yaw)

            if abs(self.headpose['roll_angle']) > 10:
                indexFace = '0'

        self.indexFace = self.mapIndexHeadPoseDict[indexFace]
    
    def faceSuggestion(self):
        if self.indexFace == None:
            print("no indexFace")
        else:
            numFace = self.mapIndexHeadPoseDict.keys()[self.mapIndexHeadPoseDict.values().index(self.indexFace)] 
            
        if numFace == '1':
            self.comment = [-1,1]
        elif numFace == '2':
            self.comment = [0,1]
        elif numFace == '3':
            self.comment = [1,1]
        elif numFace == '4':
            self.comment = [-1,0]
        elif numFace == '6':
            self.comment = [1,0]
        elif numFace == '7':
            self.comment = [-1,-1]
        elif numFace == '8':
            self.comment = [0,-1]
        elif numFace == '9':
            self.comment = [1,-1]
        elif numFace == '5':
            self.comment = [0,0]            







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

import json
from facepp import Facepp

class Face:

    def __init__(self, rep, identity=None, cluster=None, phash=None, content=None, label=None, bbox=None, facepp=Facepp()):
        self.rep = rep
        self.identity = identity
        self.cluster = cluster
        self.phash = phash
        self.content = content
        self.label = label
        self.bbox = bbox
        self.facepp = facepp
        self.faceComment = ""

    def convertToJson(self):
        jsonObj = json.dumps(self, default=lambda o: o.__dict__)
        print("jsonObj - {}".format(jsonObj))
        return jsonObj

    def __repr__(self):
        return "{{id: {}, rep[0:5]: {}, cluster:{}, phash:{}, content:{}, label:{}}}".format(
            str(self.identity),
            self.rep[0:5],
            str(self.cluster),
            self.phash,
            self.content,
            self.label
        )

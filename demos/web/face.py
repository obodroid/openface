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


class Face:

    def __init__(self, rep, identity, phash=None, content=None, name=None):
        self.rep = rep
        self.identity = identity
        self.phash = phash
        self.content = content
        self.name = name

    def convertToJson(self):
        jsonObj = json.dumps(self, default=lambda o: o.__dict__)
        print("jsonObj - {}".format(jsonObj))
        return jsonObj

    def __repr__(self):
        return "{{id: {}, rep[0:5]: {}, phash:{}, content:{}, name:{}}}".format(
            str(self.identity),
            self.rep[0:5],
            self.phash,
            self.content,
            self.name
        )


def convertToFace(rep, identity, phash=None, content=None, name=None):
    return Face(rep, identity, phash, content, name)

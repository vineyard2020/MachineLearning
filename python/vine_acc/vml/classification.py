#
# Copyright 2018 Vineyard
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#		http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import jnius_config
jnius_config.add_options('-Xss1g', '-Xmx1g')

from jnius import autoclass
import numpy as np

JString = autoclass('java.lang.String')

class LogisticRegressionModel(object):

    def __init__(self, numClasses = 10, numFeatures = 784):

        LRModel = autoclass('org.vineyard.vml.java.LogisticRegressionModel')
        self.LR = LRModel(numClasses, numFeatures)

    def getIntercepts(self):

        return np.asarray(self.LR.getIntercepts())

    def getWeights(self):

        return np.asarray(self.LR.getWeights())

    def getNumClasses(self):

        return self.LR.getNumClasses()

    def getNumFeatures(self):

        return self.LR.getNumFeatures()

    def load(self, filename):

        self.LR.load(JString(filename))

    def test(self, filename):

        self.LR.test(JString(filename))

    def train(self, filename, numExamples, alpha, gamma, numIterations):

        self.LR.train(JString(filename), int(numExamples), float(alpha), float(gamma), int(numIterations))

    def trainCPU(self, filename, numExamples, alpha, gamma, numIterations):

        self.LR.trainCPU(JString(filename), int(numExamples), float(alpha), float(gamma), int(numIterations))

    def save(self, filename):

        self.LR.save(JString(filename))

    def setIntercepts(self, weights):

        self.LR.setIntercepts(weights.flatten().astype(float).tolist())

    def setWeights(self, weights):

        self.LR.setWeights(weights.flatten().astype(float).tolist())

    def setNumClasses(self, numClasses):

        self.LR.setNumClasses(int(numClasses))

    def setNumFeatures(self, numFeatures):

        self.LR.setNumFeatures(int(numFeatures))

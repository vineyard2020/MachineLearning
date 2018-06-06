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

class KMeansModel(object):

    def __init__(self, numClusters = 14, numFeatures = 784):

        KMModel = autoclass('org.vineyard.vml.java.KMeansModel')
        self.KM = KMModel(numClusters, numFeatures)

    def getCentroids(self):

        return np.asarray(self.KM.getCentroids())

    def getNumClusters(self):

        return self.KM.getNumClusters()

    def getNumFeatures(self):

        return self.KM.getNumFeatures()

    def load(self, filename):

        self.KM.load(JString(filename))

    def test(self, filename, numClasses):

        self.KM.test(JString(filename), int(numClasses))

    def train(self, filename, numExamples, numIterations):

        self.KM.train(JString(filename), int(numExamples), int(numIterations))

    def trainCPU(self, filename, numExamples, numIterations):

        self.KM.trainCPU(JString(filename), int(numExamples), int(numIterations))

    def save(self, filename):

        self.KM.save(JString(filename))

    def setCentroids(self, centroids):

        self.KM.setCentroids(centroids.flatten().astype(float).tolist())

    def setNumClusters(self, numClusters):

        self.KM.setNumClusters(int(numClusters))

    def setNumFeatures(self, numFeatures):

        self.KM.setNumFeatures(int(numFeatures))

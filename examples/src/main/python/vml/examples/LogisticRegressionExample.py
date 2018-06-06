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

from vml.classification import LogisticRegressionModel
from sys import argv

import numpy as np


if __name__ == "__main__":

    if len(argv) != 6:
        print("Usage: LogisticRegressionApp <dataset> <numExamples> <alpha> <gamma> <numIterations>")
        exit(-1)

    dataset = argv[1]
    numExamples = int(argv[2])
    alpha = float(argv[3])
    gamma = float(argv[4])
    numIterations = int(argv[5])

    train_dataset = "/home/centos/vineyard/data/" + dataset + "_train.dat"
    test_dataset = "/home/centos/vineyard/data/" + dataset + "_test.dat"

    with open("/home/centos/vineyard/data/" + dataset, 'r') as f:
        for line in f:
            if line[0] != '#':
                parameters = line.split(',')
                numClasses = int(parameters[0])
                numFeatures = int(parameters[1])
    f.close()

    LR = LogisticRegressionModel(numClasses, numFeatures)
    LR.train(train_dataset, numExamples, alpha, gamma, numIterations)
    LR.test(test_dataset)

    weights = np.zeros((numClasses, numFeatures)).astype(float)
    intercepts = np.zeros((numClasses)).astype(float)

    LR.setWeights(weights);
    LR.setIntercepts(intercepts);

    LR.trainCPU(train_dataset, numExamples, alpha, gamma, numIterations)
    LR.test(test_dataset)

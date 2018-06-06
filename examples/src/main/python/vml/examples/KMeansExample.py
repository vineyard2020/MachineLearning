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

from vml.clustering import KMeansModel
from sys import argv

if __name__ == "__main__":

    if len(argv) != 5:
        print("Usage: KMeansApp <dataset> <numExamples> <numClusters> <numIterations>")
        exit(-1)

    dataset = argv[1]
    numExamples = int(argv[2])
    numClusters = int(argv[3])
    numIterations = int(argv[4])

    train_dataset = "/home/centos/vineyard/data/" + dataset + "_train.dat_noLabels"
    test_dataset = "/home/centos/vineyard/data/" + dataset + "_test.dat"

    with open("/home/centos/vineyard/data/" + dataset, 'r') as f:
        for line in f:
            if line[0] != '#':
                parameters = line.split(',')
                numClasses = int(parameters[0])
                numFeatures = int(parameters[1])
    f.close()

    KM = KMeansModel(numClusters, numFeatures)
    KM.train(train_dataset, numExamples, numIterations)
    KM.test(test_dataset, numClasses)

    KM2 = KMeansModel(numClusters, numFeatures)
    KM2.trainCPU(train_dataset, numExamples, numIterations)
    KM2.test(test_dataset, numClasses)

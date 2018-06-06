/*
 * Copyright 2018 Vineyard
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *		http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cstdlib>
#include <cstdio>
#include <string>
#include <fstream>
#include <iostream>

#include "/home/centos/vineyard/lib/libVine.h"

using namespace std;

int main(int argc, char **argv){

	if (argc != 5){
		cout << "Usage: " << argv[0] <<" <dataset> <numExamples> <numClusters> <numIterations>" << endl;
		exit(-1);
	}

	string dataset = argv[1];
	int numExamples = atoi(argv[2]);
	int numClusters = atoi(argv[3]);
	int numIterations = atoi(argv[4]);
	int numClasses = 0;
	int numFeatures = 0;

	string train_dataset = "/home/centos/vineyard/data/" + dataset + "_train.dat_noLabels";
	string test_dataset = "/home/centos/vineyard/data/" + dataset + "_test.dat";

	ifstream config_file;
	config_file.open(("/home/centos/vineyard/data/" + dataset).c_str());

	string line;
	string delimiter = ", ";

	if (config_file.is_open()) {
		getline(config_file, line);
		size_t pos = line.find_first_of(delimiter, 0);
		numClasses = stoi(line.substr(0, pos));
		line.erase(0, pos + 1);
		numFeatures = stoi(line.substr(0, string::npos));
		line.erase(0, string::npos);
		config_file.close();
	}
	else {
		cerr << "Error opening config_file!" << endl;
		exit(1);
	}

	KMeansModel KM;
	KM.setNumClusters(numClusters);
	KM.train(train_dataset, numExamples, numIterations);
	KM.test(test_dataset, numClasses);

	KMeansModel KM2;
	KM.setNumClusters(numClusters);
	KM2.trainCPU(train_dataset, numExamples, numIterations);
	KM2.test(test_dataset, numClasses);

	return 0;
}

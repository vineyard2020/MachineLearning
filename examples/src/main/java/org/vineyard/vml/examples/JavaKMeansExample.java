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

package org.vineyard.vml.examples;

import org.vineyard.vml.java.KMeansModel;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileNotFoundException;
import java.io.IOException;

class JavaKMeansExample{

	public static void main(String[] args){

		String dataset = "";
		int numIterations = 0;
		int numClusters = 0;
		int numExamples = 0;
		int numClasses = 0;
		int numFeatures = 0;

		if (args.length == 4) {
    			try {
				dataset = args[0];
				numExamples = Integer.parseInt(args[1]);
				numClusters = Integer.parseInt(args[2]);
				numIterations = Integer.parseInt(args[3]);
			} catch (NumberFormatException e) {
				System.err.println("Arguments " + args[1] + ", " + args[2] + ", " +args[3]+ " must be integers.");
				System.exit(1);
			}
		}
		else{
			System.out.println("Dataset, numExamples, alpha and numIterations needed.");
			System.out.println("Usage: java KMeansApp <dataset> <numExamples> <numClusters> <numIterations>");
			System.exit(2);
		}

		File config_file = new File("/home/centos/vineyard/data/" + dataset);
		BufferedReader reader = null;

		try {
			reader = new BufferedReader(new FileReader(config_file));
			String line = null;

			while ((line = reader.readLine()) != null) {
				String[] parameters = line.split(",");
				numClasses = Integer.parseInt(parameters[0].trim());
				numFeatures = Integer.parseInt(parameters[1].trim());
			}
		}
		catch (FileNotFoundException e) {
			e.printStackTrace();
			System.exit(1);
		}
		catch (IOException e) {
			e.printStackTrace();
			System.exit(1);
		}
		finally {
			try {
				if (reader != null) {
					reader.close();
				}
			}
			catch (IOException e) {
				System.out.println("Can't close configuration file.");
				e.printStackTrace();
				System.exit(1);
			}
		}

		KMeansModel KM = new KMeansModel();
		KM.setNumClusters(numClusters);
		KM.train("/home/centos/vineyard/data/" + dataset + "_train.dat_noLabels", numExamples, numIterations);
		KM.test("/home/centos/vineyard/data/" + dataset + "_test.dat", numClasses);

		KMeansModel KM2 = new KMeansModel();
		KM2.setNumClusters(numClusters);
		KM2.trainCPU("/home/centos/vineyard/data/" + dataset + "_train.dat_noLabels", numExamples, numIterations);
		KM2.test("/home/centos/vineyard/data/" + dataset + "_test.dat", numClasses);
	}

}

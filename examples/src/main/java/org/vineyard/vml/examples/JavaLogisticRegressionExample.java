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

import org.vineyard.vml.java.LogisticRegressionModel;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileNotFoundException;
import java.io.IOException;

class JavaLogisticRegressionExample{

	public static void main(String[] args){

		String dataset = "";
		float alpha = (float) 0.5;
		float gamma = (float) 0.9;
		int numIterations = 0;
		int numExamples = 0;
		int numClasses = 0;
		int numFeatures = 0;

		if (args.length == 5) {
    			try {
				dataset = args[0];
				numExamples = Integer.parseInt(args[1]);
				alpha = Float.parseFloat(args[2]);
				gamma = Float.parseFloat(args[3]);
				numIterations = Integer.parseInt(args[4]);
			} catch (NumberFormatException e) {
				System.err.println("Arguments " + args[2] + ", " + args[3] + " must be float and arguments " + args[1] + ", " + args[4] + " must be integers.");
				System.exit(1);
			}
		}
		else{
			System.out.println("Dataset, numExamples, alpha and numIterations needed.");
			System.out.println("Usage: java LogisticRegressionApp <dataset> <numExamples> <alpha> <gamma> <numIterations>");
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

		LogisticRegressionModel LR = new LogisticRegressionModel(numClasses, numFeatures);
		LR.train("/home/centos/vineyard/data/" + dataset + "_train.dat", numExamples, alpha, gamma, numIterations);
		LR.test("/home/centos/vineyard/data/" + dataset + "_test.dat");

		float[] weights = new float[numClasses * numFeatures];
		float[] intercepts = new float[numClasses];

		LR.setWeights(weights);
		LR.setIntercepts(intercepts);

		LR.trainCPU("/home/centos/vineyard/data/" + dataset + "_train.dat", numExamples, alpha, gamma, numIterations);
		LR.test("/home/centos/vineyard/data/" + dataset + "_test.dat");

	}
}

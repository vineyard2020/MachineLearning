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

package org.vineyard.vml.examples

import org.vineyard.vml.scala.LogisticRegressionModel

import scala.io.Source

object LogisticRegressionExample{

	def main(args: Array[String]){

		var dataset: String = null;
		var alpha: Float = 0.5.toFloat;
		var gamma: Float = 0.9.toFloat;
		var numIterations: Int = 0;
		var numExamples: Int = 0;

		if (args.length == 5) {
			try {
				dataset = args(0);
				numExamples = args(1).toInt;
				alpha = args(2).toFloat;
				gamma = args(3).toFloat;
				numIterations = args(4).toInt;
			}
			catch {
				case e @ ( _ : NumberFormatException) => {
					System.err.println("Arguments " + args(2) + ", " + args(3) + " must be float and arguments " + args(1) + ", " + args(4) + " must be integers.");
					System.exit(1);
				}
			}
		}
		else{
			System.err.println("Dataset, numExamples, alpha and numIterations needed.");
			System.err.println("Usage: scala LogisticRegressionApp <dataset> <numExamples> <alpha> <gamma> <numIterations>");
			System.exit(2);
		}

		val config_file = Source.fromFile("/home/centos/vineyard/data/" + dataset)
		val parameters = try config_file.getLines.mkString.split(",") finally config_file.close()
		val numClasses = parameters(0).toInt
		val numFeatures = parameters(1).toInt

		val LR = new LogisticRegressionModel();
		LR.setNumClasses(numClasses)
		LR.train("/home/centos/vineyard/data/" + dataset + "_train.dat", numExamples, alpha, gamma, numIterations);
		LR.test("/home/centos/vineyard/data/" + dataset + "_test.dat");

		var weights: Array[Float] = new Array[Float] (numClasses * numFeatures);
		var intercepts: Array[Float] = new Array[Float] (numClasses);

		LR.setWeights(weights);
		LR.setIntercepts(intercepts);

		LR.trainCPU("/home/centos/vineyard/data/" + dataset + "_train.dat", numExamples, alpha, gamma, numIterations);
		LR.test("/home/centos/vineyard/data/" + dataset + "_test.dat");

	}
}

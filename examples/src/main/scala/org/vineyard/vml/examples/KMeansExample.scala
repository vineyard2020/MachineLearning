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

import org.vineyard.vml.scala.KMeansModel

import scala.io.Source

object KMeansExample{

	def main(args: Array[String]){

		var dataset: String = null
		var numClusters: Int = 0
		var numIterations: Int = 0
		var numExamples: Int = 0

		if (args.length == 4) {
			try {
				dataset = args(0)
				numExamples = args(1).toInt
				numClusters = args(2).toInt
				numIterations = args(3).toInt
			}
			catch {
				case e @ ( _ : NumberFormatException) => {
					System.err.println("Arguments " + args(1) + ", " + args(2) + ", " + args(3) + " must be integers.")
					System.exit(1)
				}

			}
		}
		else {
			System.err.println("Dataset, numExamples, alpha and numIterations needed.")
			System.err.println("Usage: scala KMeansApp <dataset> <numExamples> <numClusters> <numIterations>")
			System.exit(2)
		}

		val config_file = Source.fromFile("/home/centos/vineyard/data/" + dataset)
		val parameters = try config_file.getLines.mkString.split(",") finally config_file.close()
		val numClasses = parameters(0).toInt
		val numFeatures = parameters(1).toInt

		val KM = new KMeansModel()
		KM.setNumClusters(numClusters)
		KM.train("/home/centos/vineyard/data/" + dataset + "_train.dat_noLabels", numExamples, numIterations)
		KM.test("/home/centos/vineyard/data/" + dataset + "_test.dat", numClasses)

		val KM2 = new KMeansModel()
		KM2.setNumClusters(numClusters)
		KM2.trainCPU("/home/centos/vineyard/data/" + dataset + "_train.dat_noLabels", numExamples, numIterations)
		KM2.test("/home/centos/vineyard/data/" + dataset + "_test.dat", numClasses)
	}
}

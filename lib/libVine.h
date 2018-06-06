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

using namespace std;

extern "C" {

	void* INalligned_malloc(size_t);

	void* INmalloc(size_t);

	long gradients_init(float *, int);

	void gradients_run(long, float *, float *, int);

	void gradients_end(long);

	long centroids_init(float *, int);

	void centroids_run(long, float *, float *, int);

	void centroids_end(long);

}

/*! \class KMeansModel
 * \brief Constructs a KMeans Model using a training dataset
 *
 * 	K-means is one of the simplest unsupervised learning algorithms that solve the well known clustering problem and is applicable in a variety of disciplines, such as computer vision, biology, and economics. It attempts to group individuals in a population together by similarity, but not driven by a specific purpose.
 *
 * The procedure follows a simple and easy way to cluster the training data points into a predefined number of clusters (\f$K\f$). The main idea is to define \f$K\f$ centroids \f$c\f$, one for each cluster.
 *
 * Given a set of \f$numExamples\f$ (\f$n\f$) observations \f$\{x^{0},x^{1},…,x^{n-1}\}\f$, where each observation is an \f$m\f$-dimensional real vector, KMeans clustering aims to partition the \f$n\f$ observations into \f$K\f$ (\f$\leq n\f$) sets \f$\{s^{0},s^{1},…,s^{K-1}\}\f$ so as to minimize total intra-cluster variance, or, the squared error function:
 * \f[J = \sum_{k=1}^{K} \sum_{x \in s^{k}} ||x - c^{k}||^2\f]
 *
 * The algorithm as described, starts with a random set of \f$K\f$ centroids (\f$c\f$). During each update step, all observations \f$x\f$ are assigned to their nearest centroid, while afterwards, these center points are repositioned by calculating the mean of the assigned observations to the respective centroids.
 */
class KMeansModel{

	private:
	float *centroids;

	int numClusters;
	int numFeatures;

	void centroidsKernelSw(float *data, float *countsSums, int numExamples);

	void checkLabel(int, int);

	void errorClusters(int);

	void errorFeatures(int);

	int predict(float *features);

	public:

		//!A constructor taking no arguments .
		/*!
			Calling this constructor, a KMeansModel object is created using the accelerator's default max values for the number of clusters and features (\f$10\f$ and \f$784\f$ accordingly).

			To set the number of clusters or features to a different value, setNumClusters() and setNumFeatures() functions can be used.
		*/
		KMeansModel();

		//!A constructor taking 2 arguments for the number of clusters and features.
		/*!
			Calling this constructor, a KMeansModel object is created using the given number of clusters and features (max \f$10\f$ and \f$784\f$ accordingly).
		*/
	KMeansModel(int numClusters, int numFeatures);

	//!A destructor for a given object.
	/*!
		The destructor takes no arguments and frees any allocated memory.
	*/
	~KMeansModel();

	//! A function used to get the centroids of a model.
		 /*!
			 \param -
			 \return Returns an array (float *) containing a model's centroids.
			 \sa setCentroids()
		 */
	float *getCentroids();

	//! A function used to retrieve the number of clusters.
		 /*!
			 \param -
			 \return Returns the number of clusters using an integer value.
			 \sa setNumClusters(), getNumFeatures(), setNumFeatures()
		 */
	int getNumClusters();

	//! A function used to retrieve the number of features.
		 /*!
			 \param -
			 \return Returns the number of features using an integer value.
			 \sa setNumFeatures(), getNumClusters(), setNumClusters()
		 */
	int getNumFeatures();

	//! A function used to load a model from the disk.
		 /*!
			 \param filename Path and name of the file containing the model.
			 \return Returns nothing. The model is loaded directly to the object's centroids.
			 \sa save()
		 */
	void load(string filename);

	//! A function used to test a trained model using a test file.
		 /*!
			 \param filename Path and name of the file to test the model. If the file doesn't exist an exception is raised and the function exits.
			 \param numClasses The number of classes of the test dataset.
			 \return Returns nothing. Prints metrics for the trained model's clusters using the test file provided.
			 \sa train(), trainCPU()
		 */
	void test(string filename, int numClasses);

	//! A function used to train a model using the FPGA.
		 /*!
			 \param filename Path and name of the input dataset to train the model. If the file doesn't exist an exception is raised and the function exits.
			 \param numExamples Number of examples in the dataset used to train the file (e.g \f$4000\f$ for \f$4000\f$ of \f$data lines\f$).
			 \param numIterations Number of iterations to run the KMeans algorithm.
			 \return Returns nothing. The model is now trained. Centroids are automatically updated.
			 \sa trainCPU(), test()
		 */
	void train(string filename, int numExamples, int numIterations);

	//! A function used to train a model using the CPU.
     /*!
		   \param filename Path and name of the input dataset to train the model. If the file doesn't exist an exception is raised and the function exits.
		   \param numExamples Number of examples in the dataset used to train the file (e.g \f$4000\f$ for \f$4000\f$ of \f$data lines\f$).
		   \param numIterations Number of iterations to run the KMeans algorithm.
		   \return Returns nothing. The model is now trained. Centroids are automatically updated.
       \sa train(), test()
     */
	void trainCPU(string filename, int numExamples, int numIterations);

	//! A function used to save a model (centroids) to the disk.
		 /*!
			 \param filename Path and name of the file to write the model. If the file doesn't exist it is created, otherwise it overwrites any data in the file.
			 \return Returns nothing.
			 \sa load()
		 */
	void save(string filename);

	//! A function used to set the centroids of a model.
		 /*!
			 \param centroids An array (float *) of centroids to be loaded to the model.
			 \return Returns nothing.
			 \sa getCentroids()
		 */
	void setCentroids(float *centroids);

	//! A function used to set the number of clusters.
		 /*!
			 \param numClusters The number of a model's clusters.
			 \return Returns nothing.
			 \sa getNumClusters(), getNumFeatures(), setNumFeatures()
		 */
	void setNumClusters(int numClusters);

	//! A function used to set the number of features.
		 /*!
			 \param numFeatures The number of a model's features.
			 \return Returns nothing.
			 \sa getNumFeatures(), getNumClusters(), setNumClusters()
		 */
	void setNumFeatures(int numFeatures);

};

/*! \class LogisticRegressionModel
 * \brief Constructs a Logistic Regression Model using Gradient Descent (GD) over the training set.
 *
 * Logistic Regression is used for building predictive models for many complex pattern-matching and classification problems. It is used widely in such diverse areas as bioinformatics, finance and data analytics. It is also one of the most popular machine learning techniques. It belongs to the family of classifiers known as the exponential or log-linear classifiers and is widely used to predict a binary response.
 *
 * For binary classification problems, the algorithm outputs a binary logistic regression model. Given a new data point, denoted by \f$x\f$, where \f$x_{0}=1\f$ is the intercept term, the model makes predictions by applying the logistic function \f$h(z)= \frac {1}{1+e^{-z}}\f$, where \f$z=w^{T}x\f$.
 *
 * By default, if \f$h(w^{T}x)>0.5\f$, the outcome is positive, or negative otherwise, though unlike linear SVMs (Support Vector Machines), the raw output of the logistic regression model, \f$h(z)\f$, has a probabilistic interpretation (i.e., the probability that \f$x\f$ is positive).
 *
 * Given a training set with \f$numExamples\f$ (\f$n\f$) data points and \f$numFeatures\f$ (\f$m\f$) features (not counting the intercept term) \f$\{(x^{0},y^{0}),(x^{1},y^{1}),…,(x^{n-1},y^{n-1})\}\f$, where \f$y^{i}\f$ is the binary label for input data \f$x^{i}\f$ indicating whether it belongs to the class or not, logistic regression tries to find the parameter argument \f$w\f$ (weights) that minimizes the following cost function:
 *	\f[J(w)=- \frac {1}{n} \sum _{i=0}^{n-1}\{y^{i} \log [h(w^{T}x^{i})]+(1-y^{i}) \log [1-h(w^{T}x^{i})]\}\f]
 *
 * The problem is solved using Gradient Descent (GD) over the training set (\f$\alpha\f$ is the learning rate).
 *
 *	For multi-class classification problems, the algorithm compares every class with all the remaining classes (One versus Rest) and outputs a multinomial logistic regression model, which contains \f$numClasses\f$ (\f$k\f$) binary logistic regression models. Given a new data point, \f$k\f$ models will be run, and the class with largest probability will be chosen as the predicted class.
 */
class LogisticRegressionModel{

	private:
	float *weights;		/*!< Holds a model's weights*/
	float *intercepts;	/*!< Holds a model's intercepts*/

	int numClasses;		/*!< Holds a model's number of classes*/
	int numFeatures;	/*!< Holds a model's number of features*/

	void checkLabel(int);

	void errorClasses(int);

	void errorFeatures(int);

	void gradientsKernelSw(float *data, float *gradients, int numExamples);

	int predict(float *features);

	public:

	//!A constructor taking no arguments .
  /*!
		Calling this constructor, a LogisticRegressionModel object is created using the accelerator's default max values for the number of classes and features (\f$10\f$ and \f$784\f$ accordingly).

		To set the number of classes or features to a different value, setNumClasses() and setNumFeatures() functions can be used.
	*/
	LogisticRegressionModel();

	//!A constructor taking 2 arguments for the number of classes and features.
	/*!
		Calling this constructor, a LogisticRegressionModel object is created using the given number of classes and features (max \f$10\f$ and \f$784\f$ accordingly).
	*/
	LogisticRegressionModel(int numClasses, int numFeatures);

	//!A destructor for a given object.
	/*!
		The destructor takes no arguments and frees any allocated memory.
	*/
	~LogisticRegressionModel();

	//! A function used to get the intercepts of a model.
		 /*!
			 \param -
			 \return Returns an array (float *) containing a model's intercepts.
			 \sa setIntercepts(), getWeights(), setWeights()
		 */
	float *getIntercepts();

	//! A function used to get the weights of a model.
		 /*!
			 \param -
			 \return Returns an array (float *) containing a model's weights.
			 \sa setWeights(), getIntercepts(), setIntercepts()
		 */
	float *getWeights();

	//! A function used to retrieve the number of classes.
		 /*!
			 \param -
			 \return Returns the number of classes using an integer value.
			 \sa setNumClasses(), getNumFeatures(), setNumFeatures()
		 */
	int getNumClasses();

	//! A function used to retrieve the number of features.
		 /*!
			 \param -
			 \return Returns the number of features using an integer value.
			 \sa setNumFeatures(), getNumClasses(), setNumClasses()
		 */
	int getNumFeatures();

	//! A function used to load a model from the disk.
     /*!
       \param filename Path and name of the file containing the model.
       \return Returns nothing. The model is loaded directly to the object's weights and intercepts.
       \sa save()
     */
	void load(string filename);

	//! A function used to test a trained model using a test file.
     /*!
       \param filename Path and name of the file to test the model. If the file doesn't exist an exception is raised and the function exits.
       \return Returns nothing. Prints metrics for the accuracy of the trained model using the test file provided.
       \sa train(), trainCPU()
     */
	void test(string filename);

	//! A function used to train a model using the FPGA.
     /*!
       \param filename Path and name of the input dataset to train the model. If the file doesn't exist an exception is raised and the function exits.
			 \param numExamples Number of examples in the dataset used to train the file (e.g \f$4000\f$ for \f$4000\f$ of \f$data lines\f$).
			 \param alpha The learning rate \f$\alpha\f$.
			 \param numIterations Number of iterations of Gradient Descent (GD) to run.
       \return Returns nothing. The model is now trained. Intercepts and weights are automatically updated.
       \sa trainCPU(), test()
     */
	void train(string filename, int numExamples, float alpha, float gamma, int numIterations);

	//! A function used to train a model using the CPU.
     /*!
		 	 \param filename Path and name of the input dataset to train the model. If the file doesn't exist an exception is raised and the function exits.
		 	 \param numExamples Number of examples in the dataset used to train the file (e.g \f$4000\f$ for \f$4000\f$ of \f$data lines\f$).
		 	 \param alpha The learning rate \f$\alpha\f$.
		 	 \param numIterations Number of iterations of Gradient Descent (GD) to run.
		 	 \return Returns nothing. The model is now trained. Intercepts and weights are automatically updated.
       \sa train(), test()
     */
	void trainCPU(string filename, int numExamples, float alpha, float gamma, int numIterations);
	//! A function used to save a model (weights and intercepts) to the disk.
     /*!
       \param filename Path and name of the file to write the model. If the file doesn't exist it is created, otherwise it overwrites any data in the file.
       \return Returns nothing.
       \sa load(), weights, intercepts
     */
	void save(string filename);

	//! A function used to set the intercepts of a model.
		 /*!
			 \param intercepts An array (float *) of intercepts to be loaded to the model.
			 \return Returns nothing.
			 \sa getIntercepts(), getWeights(), setWeights()
		 */
	void setIntercepts(float *intercepts);

	//! A function used to set the number of classes.
		 /*!
			 \param numClasses The number of a model's classes.
			 \return Returns nothing.
			 \sa getNumClasses(), getNumFeatures(), setNumFeatures()
		 */
	void setNumClasses(int numClasses);

	//! A function used to set the number of features.
		 /*!
			 \param numFeatures The number of a model's features.
			 \return Returns nothing.
			 \sa getNumFeatures(), getNumClasses(), setNumClasses()
		 */
	void setNumFeatures(int numFeatures);

	//! A function used to set the weights of a model.
		 /*!
			 \param weights An array (float *) of weights to be loaded to the model.
			 \return Returns nothing.
			 \sa getWeights(), getIntercepts(), setIntercepts()
		 */
	void setWeights(float *weights);

};

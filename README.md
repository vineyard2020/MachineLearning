# Vineyard Machine Learning Library (VML)

## Description

VML is a machine learning library using FPGA accelerators to speed up the training part of ML models. The integration with the FPGA is seamless, hiding all the low level communication layers with the accelerators. This way the user needs to know nothing about FPGAs or accelerators in general. VML hosts all the required libraries to train your Logistic Regression and KMeans models using mainstream high-level APIs.

## Ease of Use

VML offers high level APIs in C++, Java, Scala and Python making it easy to write your own applications or accelerate your existing ones.

## Performance

VML can provide _**2x - 20x**_ (depending on the algorithm) speedup to the execution of machine learning (ML) algorithms making it easy to train multiple models at the same time.

## Supported Algorithms

1. **Logistic Regression** (models up to 10 classes and up to 784 features supported)
2. **KMeans** (models up to 14 clusters and up to 784 features supported)

## Tools

VML provides a plethora of tools for the ML models:

- feature extraction
- saving, loading models
- constructing and evaluating models

## Instructions

This repository holds the whole VML library along with example applications. The example applications are written in C++, Java, Scala and Python.

### Directories structure:

- **afis** directory holds all the available accelerators (you have to first create an AFI from a corresponding .xclbin bitstream file.).
- **bin** directory hosts all the scripts necessary to run the example applications.
- **data** directory hosts all the necessary datasets to run the example applications.
- **examples** directory hosts the source code of all the example applications. It also contains a pom.xml in case you want to build a new jar using Maven.
- **jars** directory hosts all the classes that serve as the Java and Scala APIs. It also hosts the compiled classes for the example applications written in Java and Scala as well as other needed components.
- **lib** directory hosts all shared libraries. It hosts the drivers for communicating with the hardware accelerators as well as the C++ API.
- **python** directory hosts the Python API for all the applications.


### Installation - First Steps
1. **Create a new instance in Amazon EC2** service using Amazon's FPGA Developer AMI.
1. **Clone aws-fpga GitHub project:** Clone the project using the following command ```git clone https://github.com/aws/aws-fpga.git``` under ```/home/centos``` directory.
1. **Clone this project** also under ```/home/centos``` directory.
1. **Create the necessary AFIs:** In this project we include two xclbin files, *centroids_kernel* and *gradients kernel*. You can click [here](www.google.com) to resolve them. In order to run the applications you first have to create the corresponding AFIs and place them under the *afis* folder. Please make sure that the output files' names are centroids_kernel.xclbin and gradients_kernel.xclbin accordingly (not centroids_kernel.awsxclbin or gradients_kernel.awsxclbin). To create any AFIs please follow the instructions on the [official repository of the AWS EC2 FPGA Hardware and Software Development Kit](https://github.com/aws/aws-fpga).
1. **Install python-pip and all the necessary packages:** Install python3-pip and then install cython, numpy and pyjnius packages. These packages are necessary to run any applications using the Python api.
1. **Install Java and set JAVA_HOME:** This project was tested using Java 8. Please make sure that after installing and setting Java you modified any affected Makefiles. In our tests ```JAVA_HOME``` was set to ```/usr/lib/jvm/jdk1.8.0```
1. **Install Scala:** in order to run any applications using the Scala API, you have to install Scala for CentOS.
1. **Set environment variables:** Edit the .bashrc file of the root user, add the following lines and then source it.
```shell
	export JAVA_HOME=/usr/lib/jvm/jdk1.8.0
	export PATH=$JAVA_HOME/bin:$PATH
	export VINEYARD_HOME=/home/centos/vineyard
	export CLASSPATH=$VINEYARD_HOME/jars/*:$CLASSPATH
	export PYTHONPATH=$VINEYARD_HOME/python/vine_acc:$PYTHONPATH
	export LD_LIBRARY_PATH=$VINEYARD_HOME/lib:$LD_LIBRARY_PATH

	source /opt/Xilinx/SDx/2017.4.rte.dyn/setup.sh
```
Note that JAVA_HOME should point to the directory of your installation.
1. **Resolve the datasets:** Using this [link](www.google.com) you can download all the datasets needed for the execution of the example applications. Make sure you place all of the files under ```/home/centos/vineyard/data``` folder.

### Running Applications
To run any applications using the FPGA accelerators you must follow the next steps:

1. Login to the f1 instance using your private key.
2. Become root.

You are ready to execute any of your applications or the provided example ones.

#### Running the Example Applications:

To execute the example applications navigate to /home/centos/vineyard/bin folder. There you can find two (2) scripts: ```km.sh```, ```lr.sh```. Executing them with the *help* argument you can see all the input arguments required. Both scripts share the same arguments order. In more detail, every script requires exactly two arguments: *programming_language* and *dataset*. The programming language refers to the corresponding API that is going to be used while the dataset referes to the dataset upon each model is going to be trained. For the *programming_language* argument the available options are *cpp*, *java*, *python* and *scala* while for the *dataset* argument, *MNIST* option is available. Execution example:

```shell
	$ ./lr.sh java MNIST
```

### DATASETS

You can find all the required datasets [here](www.google.com).

### XCLBINS

You can find all the required xclbins [here](www.google.com).

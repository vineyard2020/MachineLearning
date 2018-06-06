CPP = g++

EXAMPLES = KMeansExample LogisticRegressionExample

LIBS = -L/home/centos/vineyard/lib/ -lVine -L/opt/Xilinx/SDx/2017.4.op/lib/lnx64.o/ -lxilinxopencl

all: kmeans logistic

kmeans:
	$(CPP) -O3 -Wall -o KMeansExample KMeansExample.cpp $(LIBS) -std=c++0x

logistic:
	$(CPP) -O3 -Wall -o LogisticRegressionExample LogisticRegressionExample.cpp $(LIBS) -std=c++0x

help:
	# Define the name of the application you want to compile.

clean:
	rm -f ${EXAMPLES} sdaccel_*

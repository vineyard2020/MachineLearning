#!/bin/bash

PL=${1:-'help'}
DATASET=${2:-'MNIST'}
NUMEXAMPLES=0
ALPHA=0
GAMMA=0

if [ $PL == "help" ]
then
	echo ./lr.sh \<programming language\> \<dataset\>
	exit 1
else
	if [ $DATASET = "letters" ]
	then
		NUMEXAMPLES=124800
		ALPHA=0.2
		GAMMA=0.95
	elif [ $DATASET = "MNIST" ]
	then
		NUMEXAMPLES=40000
		ALPHA=0.75
		GAMMA=0.9
	else
		echo Invalid dataset input. Choose between \'letters\' or \'MNIST\'
		exit 1
	fi

	if [ $PL == "scala" ]
	then
		time scala -J-Xss100m -J-Xmx600m org.vineyard.vml.examples.LogisticRegressionExample $DATASET $NUMEXAMPLES $ALPHA $GAMMA 100
	elif [ $PL == "java" ]
	then
		time java -Xss1g -Xmx1g org.vineyard.vml.examples.JavaLogisticRegressionExample $DATASET $NUMEXAMPLES $ALPHA $GAMMA 100
	elif [ $PL == "python" ]
	then
		time python3 ../examples/src/main/python/vml/examples/LogisticRegressionExample.py $DATASET $NUMEXAMPLES $ALPHA $GAMMA 100
	elif [ $PL == "cpp" ]
	then
		if ! [ -e ../examples/src/main/cpp/vml/examples/LogisticRegressionExample ]
		then
			# preserve the calling directory
			_calling_dir="$(pwd)"
			cd ../examples/src/main/cpp/vml/examples
			set -e
			make -f Makefile
			cd $_calling_dir
		fi
		time  ../examples/src/main/cpp/vml/examples/LogisticRegressionExample $DATASET $NUMEXAMPLES $ALPHA $GAMMA 100
	else
		echo Invalid programming language. Choose between \'java\', \'scala\', \'python\' and \'cpp\'
		exit 1
	fi
fi

rm sdaccel_profile_summary.*

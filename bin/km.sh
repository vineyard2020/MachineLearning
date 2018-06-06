#!/bin/bash

PL=${1:-'help'}
DATASET=${2:-'MNIST'}
NUMEXAMPLES=0
NUMCLUSTERS=14

if [ $PL == "help" ]
then
	echo ./km.sh \<programming language\> \<dataset\>
	exit 1
else
	if [ $DATASET = "letters" ]
	then
		NUMEXAMPLES=124800
	elif [ $DATASET = "MNIST" ]
	then
		NUMEXAMPLES=40000
	else
		echo Invalid dataset input. Choose between \'letters\' or \'MNIST\'
		exit 1
	fi

	if [ $PL == "scala" ]
	then
		time scala -J-Xss100m -J-Xmx600m org.vineyard.vml.examples.KMeansExample $DATASET $NUMEXAMPLES $NUMCLUSTERS 100
	elif [ $PL == "java" ]
	then
		time java -Xss1g -Xmx1g org.vineyard.vml.examples.JavaKMeansExample $DATASET $NUMEXAMPLES $NUMCLUSTERS 100
	elif [ $PL == "python" ]
	then
		time python3 ../examples/src/main/python/vml/examples/KMeansExample.py $DATASET $NUMEXAMPLES $NUMCLUSTERS 100
	elif [ $PL == "cpp" ]
	then
		if ! [ -e ../examples/src/main/cpp/vml/examples/KMeansExample ]
		then
			# preserve the calling directory
			_calling_dir="$(pwd)"
			cd ../examples/src/main/cpp/vml/examples
			set -e
			make -f Makefile
			cd $_calling_dir
		fi
		time  ../examples/src/main/cpp/vml/examples/KMeansExample $DATASET $NUMEXAMPLES $NUMCLUSTERS 100
	else
		echo Invalid programming language. Choose between \'java\', \'scala\', \'python\' and \'cpp\'
		exit 1
	fi
fi

rm sdaccel_profile_summary.*

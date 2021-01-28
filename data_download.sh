#!/usr/bin/env bash

set -eo pipefail

DIR="$(dirname "${BASH_SOURCE[0]}")"

DATA_IN="$DIR"/data/input

get_data() {
	mkdir -p $DATA_IN
	
	if [ -f "$DATA_IN/test_data.txt" ] && [ -f "$DATA_IN/train_neg.txt" ] && [ -f "$DATA_IN/train_pos.txt" ] && [ -f "$DATA_IN/train_neg_full.txt" ] && [ -f "$DATA_IN/train_pos_full.txt" ]; then
		echo "Twitter dataset already present, skipping download"
	elif [ -f twitter-datasets.zip ]; then
		echo "Dataset zip file already present"
		echo "Unzip to $DATA_IN"
		echo "Note: they should not be in a subfolder in $DATA_IN"
		echo "E.g. $DATA_IN/train_neg.txt is correct"
		echo "$DATA_IN/twitter-datasets/train_neg.txt is incorrect"
	else
		wget http://www.da.inf.ethz.ch/files/twitter-datasets.zip
	fi
	
	if [ -f glove.twitter.27B.zip ]; then
		echo "GloVe zip file already present"
		echo "Unzip glove.twitter.27B.zip to $DATA_IN"
	elif [[ ! -f "$DATA_IN/glove.twitter.27B.200d.txt" || ! -f "$DATA_IN/glove.twitter.27B.50d.txt" ]]; then
		echo "Downloading pretrained glove model"
		wget http://nlp.stanford.edu/data/glove.twitter.27B.zip
	else
		echo "GloVe model already present, skipping download"
		sleep 3
	fi
	
	if [ -f uncased_L-24_H-1024_A-16.zip ]; then
		echo "Bert zip file already present"
		echo "Unzip uncased_L-24_H-1024_A-16.zip to $DATA_IN"
	elif [ ! -f "$DATA_IN/uncased_L-24_H-1024_A-16/bert_model.ckpt.data-00000-of-00001" ]; then
		echo "Downloading bert model"
		wget https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-24_H-1024_A-16.zip
	else
		echo "bert model already present, skipping download"
		sleep 3
	fi
	
	if [[ "$OSTYPE" == "linux-gnu"* ]]; then
		if [[ ! -f "$DATA_IN/test_data.txt" || ! -f "$DATA_IN/train_neg.txt" || ! -f "$DATA_IN/train_pos.txt" || ! -f "$DATA_IN/train_neg_full.txt" || ! -f "$DATA_IN/train_pos_full.txt" ]] && [ -f twitter-datasets.zip ]; then
			unzip -j twitter-datasets.zip 'twitter-datasets/*' -d "$DATA_IN" 
			rm twitter-datasets.zip
		fi
		
		#code from SLT course exercise files build_vocab.sh and cut_vocab.sh
		if [ ! -f "$DATA_IN/glove_vocab.txt" ]; then
			echo "Creating vocab for self-trained glove"
			echo "Could take a minute or two"
			cat "$DATA_IN/train_pos.txt" "$DATA_IN/train_neg.txt" "$DATA_IN/test_data.txt" | sed "s/ /\n/g" | grep -v "^\s*$" | sort | uniq -c > "$DATA_IN/vocab.txt"
			cat "$DATA_IN/vocab.txt" | sed "s/^\s\+//g" | sort -rn | grep -v "^[1234]\s" | cut -d' ' -f2 > "$DATA_IN/glove_vocab.txt"
			rm "$DATA_IN/vocab.txt"
		fi
		
		if [ -f glove.twitter.27B.zip ] && [[ ! -f "$DATA_IN/glove.twitter.27B.200d.txt" || ! -f "$DATA_IN/glove.twitter.27B.50d.txt" ]]; then
			unzip glove.twitter.27B.zip -d "$DATA_IN"
			rm glove.twitter.27B.zip
			rm -f "$DATA_IN/glove.twitter.27B.25d.txt"
			rm "$DATA_IN/glove.twitter.27B.100d.txt"
		fi
		
		if [ -f uncased_L-24_H-1024_A-16.zip ] && [ ! -f "$DATA_IN/uncased_L-24_H-1024_A-16/bert_model.ckpt.data-00000-of-00001" ]; then
			unzip uncased_L-24_H-1024_A-16.zip -d "$DATA_IN"
			rm uncased_L-24_H-1024_A-16.zip
		fi

	else
		if [[ -f "$DATA_IN/test_data.txt" && -f "$DATA_IN/train_neg_full.txt" && -f "$DATA_IN/train_pos_full.txt" && ! -f "$DATA_IN/glove_vocab.txt" ]]; then
			echo "Creating vocab for self-trained glove"
			echo "Could take a minute or two"
			#code from SLT course exercise files build_vocab.sh and cut_vocab.sh
			cat "$DATA_IN/train_pos.txt" "$DATA_IN/train_neg.txt" "$DATA_IN/test_data.txt" | sed "s/ /\n/g" | grep -v "^\s*$" | sort | uniq -c > "$DATA_IN/vocab.txt"
			cat "$DATA_IN/vocab.txt" | sed "s/^\s\+//g" | sort -rn | grep -v "^[1234]\s" | cut -d' ' -f2 > "$DATA_IN/glove_vocab.txt"
		else
			echo "Must manually unzip downloaded files to $DATA_IN"
			echo "Note: twitter-datasets.zip files should not be in a subfolder in $DATA_IN"
			echo "E.g. $DATA_IN/train_neg.txt is correct"
			echo "$DATA_IN/twitter-datasets/train_neg.txt is incorrect"
			echo " "
			echo "The bert model should however be in the subfolder provided by the zip file"
			echo "E.g. $DATA_IN/uncased_L-24_H-1024_A-16/bert_model.ckpt.data-00000-of-00001"
			echo "IMPORTANT:"
			echo "If you plan to run baseline_glove models,"
			echo "re-run this script after unzipping the twitter datasets, to extract the vocabulary necessary"
			sleep 5
		fi
	fi
}

install_local_elmo() {
	virtualenv elmoenv
	source elmoenv/bin/activate
	pip3 install -r requirements_elmo_local.txt
}

install_local() {
	virtualenv normal_env
	source normal_env/bin/activate
	pip3 install -r requirements_local.txt
}

install_cluster_elmo() {
	pip install --user -r requirements_cluster.txt
}
install_cluster() {	
	pip install --user -r requirements_cluster.txt
}

usage() {
	echo "With no arguments this will download the twitter datasets."
	echo "Note: must be connected to ETH VPN to download dataset!"
	echo "If on a unix system it will also unpack it in the appropriate location."
	echo "Usage:"
	echo "-c cluster            Load install requirements for Leonhard cluster / NOTE: must still manually load modules (see readme)"
	echo "-c local              Create virtualenvironments and install requirements within"
	echo "-e                    Install requirements for elmo models (only in conjunction with -c)"
	echo "-h                    Display this message"
	sleep 5
	exit 0
}

main() {

	if [ "$cluster" = "cluster" ]; then
		if [ "$elmo" = "True" ]; then
			echo "Installing requirements for running elmo models on Leonhard"
			sleep 2
			install_cluster_elmo
		else
			echo "Installing requirements for running non-elmo models on Leonhard"
			install_cluster
		fi
	fi
	
	if [ "$cluster" = "local" ]; then
	echo "Must have virtualenv installed."
	echo "Run \"pip3 install virtualenv\" in case of failure"
		if [ "$elmo" = "True" ]; then
			echo "Installing requirements for running elmo models locally"
			echo "Using virtualenv elmoenv"
			install_local_elmo
		else
			echo "Installing requirements for running non-elmo models locally"
			echo "Using virtualenv normal_env"
			install_local
		fi
	fi
	
	
	echo "Downloading dataset"
	sleep 2
	get_data
	echo "Downloads complete"
}
while getopts :c:eh flag
do
	case "${flag}" in
		e) elmo="True";;
		c) cluster=${OPTARG};;
		h) usage;;
		\?) usage;;
	esac
done
if [ "$elmo" = "True" ] && [ -z "$cluster" ]; then
	echo "Must specify -c parameter to local or cluster when using -e"
	sleep 5
	exit 1
fi

if [ "$cluster" != "local" ] && [ "$cluster" != "cluster" ] && [ -n "$cluster" ]; then
	echo "-c argument must be either \"local\" or \"cluster\" "
	sleep 5
	exit 1
fi
main
#!/usr/bin/env bash
#
#  Execute from the current working directory
#$ -cwd
#
#$ -l short

export PATH="/home/jklee/anaconda2/bin:$PATH"
export PYTHONPATH=/home/jklee/anaconda2/lib/python2.7
source activate tensorflow

python test.py

#!/usr/bin/env bash
#
#  Execute from the current working directory
#$ -cwd
#
#$ -l short

export PATH="/home/jklee/anaconda2/bin:$PATH"
export PYTHONPATH=/home/jklee/anaconda2/lib/python2.7
source activate tensorflow

#export CUDA_HOME=/home/jklee/cuda-8.0
#export LIBRARY_PATH=$LIBRARY_PATH:${CUDA_HOME}/lib64 
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${CUDA_HOME}/lib64 
#export CPATH=${CUDA_HOME}/include:${CPATH} 
#export PATH=${CUDA_HOME}/bin:${PATH} 
#export THEANO_FLAGS=device=cuda,floatX=float32,optimizer_including=cudnn,cuda.root=${CUDA_HOME}

#echo $LD_LIBRARY_PATH
#echo $LIBRARY_PATH
#echo $CPATH
#echo $PATH

#nvidia-smi
#ls -l /usr/local
IND=$SGE_TASK_ID
echo "Portion ($PORTION) - Epsilon ($EPSILON) - ID ($SGE_TASK_ID)"
python train_krl_dqn.py ${PORTION} ${EPSILON} ${IND}

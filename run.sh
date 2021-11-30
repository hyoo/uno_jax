# export XLA_PYTHON_CLIENT_PREALLOCATE=false
# export CUDA_VISIBLE_DEVICES=0
python train.py -e 5 --lr 4e-4 -z 512

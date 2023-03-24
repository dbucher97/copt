#! /usr/bin/env sh

poetry run pip install --upgrade "jax[cuda11_cudnn82]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/lib/python3/dist-packages/tensorflow/"
export LIBRARY_PATH="$LIBRARY_PATH:/usr/lib/python3/dist-packages/tensorflow/"

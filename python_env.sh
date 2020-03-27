#!/usr/bin/env bash

# Build Python virtualenv for tensorflow.
if [[ ! -d ./python_env ]]; then
    virtualenv -p /usr/bin/python3.5m python_env
fi

source ./python_env/bin/activate
pip install --upgrade virtualenv -i https://pypi.tuna.tsinghua.edu.cn/simple/
# wget -O ./python_env/tensorflow-1.8.0-cp35-cp35m-linux_x86_64.whl https://cloud.tsinghua.edu.cn/f/8535b181129249caad46/?dl=1
# pip install ./python_env/tensorflow-1.8.0-cp35-cp35m-linux_x86_64.whl
pip install tensorflow==1.3.0 -i https://pypi.tuna.tsinghua.edu.cn/simple/
pip install scikit-image -i https://pypi.tuna.tsinghua.edu.cn/simple/
pip install keras==2.0.8 -i https://pypi.tuna.tsinghua.edu.cn/simple/
pip install IPython -i https://pypi.tuna.tsinghua.edu.cn/simple/
pip install h5py -i https://pypi.tuna.tsinghua.edu.cn/simple/
pip install cython -i https://pypi.tuna.tsinghua.edu.cn/simple/
pip install imgaug -i https://pypi.tuna.tsinghua.edu.cn/simple/
pip install opencv-python -i https://pypi.tuna.tsinghua.edu.cn/simple/
pip install pytoml -i https://pypi.tuna.tsinghua.edu.cn/simple/
pip install pycocotools -i https://pypi.tuna.tsinghua.edu.cn/simple/
#pip install python3-tk -i https://pypi.tuna.tsinghua.edu.cn/simple/

# Provide numpy headers to C++
ln -s `python -c "import numpy as np; print(np.__path__[0])"`/core/include/numpy python_env/ || true

deactivate

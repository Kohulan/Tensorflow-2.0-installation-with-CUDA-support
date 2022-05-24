# CUDA-11-with-Tensoflow2.0-Installation-Guide

Installing Nvidia Drivers, Installing CUDA drivers with cuDNN on a Ubuntu machine is not straightforward. Where many tutorials give a detail step-by-step guide to install Tensorflow-1.0, There is no proper tutorial which explains the steps a beginner should take when installing Tensorflow-2.0. This is a more elaborative guide on installing All the necessary drivers and kick off your first machine learning algorithm.

## First, remove all previous CUDA and NVIDIA installation.
```shell
sudo apt-get --purge remove "*cublas*" "cuda*" "nsight*" "*nvidia*"
sudo nano /etc/apt/sources.list #comment nvidia dev 
sudo apt --fix-broken install
```
## Second, add NVIDIA package repositories: 
```shell
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin 
sudo mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600 
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
```
You might need to check the https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/ link to check for the latest keys pub file and modify the previous line.

```shell
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/ /" 
sudo apt-get update 
wget http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb 
sudo apt install ./nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb 
sudo apt-get update 
wget --no-check-certificate https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/libnvinfer7_7.1.3-1+cuda11.0_amd64.deb 
sudo apt install ./libnvinfer7_7.1.3-1+cuda11.0_amd64.deb 
sudo apt-get update 
```

## Third, install development and runtime libraries (~4GB)
```shell
sudo apt-get install --no-install-recommends cuda-11-3 libcudnn8=8.2.1.32-1+cuda11.3 libcudnn8-dev=8.2.1.32-1+cuda11.3 #cuda-runtime-11-3 cuda-demo-suite-11-3 cuda-drivers-510 nvidia-driver-510 libnvidia-extra-510
sudo apt-get update 
```
## Finally, reboot the PC and check the installation
```shell
sudo reboot
nvidia-smi
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 515.43.04    Driver Version: 515.43.04    CUDA Version: 11.7     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA GeForce ...  On   | 00000000:3B:00.0 Off |                  N/A |
| 23%   27C    P8    16W / 250W |      1MiB / 11264MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
```
## Installing pip3 and Tensorflow-2.x-GPU

  - The support for python v2.7 ended officially in 2020, So it's better if we can stick with python version 3.6

### Step 1 (Installing pip3):
  - Use the following commend to install pip3 in your PC,
  ```shell
  $ sudo apt-get install python3-pip
  $ sudo pip3 --upgrade pip
  ```
  
### Step 2 (Installing Tensorflow):
  - Now let's install Tensorflow 2.x
  ```shell
  $ pip3 install tensorflow-gpu==2.x.0
  ```
### Step 3 (Verifying the installation):
  - Run the following inside python3 terminal to verify the installation
  ```shell
  $ python3
  ```
  ```python
  >>> import tensorflow as tf
  >>> hello = tf.constant('hello tensorflow')
  >>> x = [[2.]]
  >>> print('hello, {}'.format(tf.matmul(x, x)))
  2019-00-00 16:04:38.589080: I tensorflow/stream_executor/platform/default/dso_loader.cc:42] Successfully opened dynamic library   libcublas.so.10.0
  hello, [[4.]]
  >>> exit()
  ```
### That's it you have successfully Tensorflow-2.x-GPU with CUDA 11.0.
  

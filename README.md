# CUDA-10-with-Tensoflow2.0-Installation-Guide

Installing Nvidia Drivers, Installing CUDA drivers with cuDNN on a Ubuntu machine is not straightforward. Where many tutorials give a detail step-by-step guide to install Tensorflow-1.0, There is no proper tutorial which explains the steps a beginner should take when installing Tensorflow-2.0. This is a more elaborative guide on installing All the necessary drivers and kick off your first machine learning algorithm.

## First things first (Check your nVidia Device)

If you know which nVidia graphics card you have in your machine you can continue to step 2, otherwise, you can use the following script to check whether you have an nVidia graphics card and its type.
```shell_session
$ lspci | grep -i nvidia
```
### Step 1: 
  - After figuring out your nVidia graphics card you can visit: https://www.nvidia.com/Download/index.aspx?lang=en-us, to know which version of the driver supports your hardware.
  - In my case I am using an nVidia Tesla V100, So, my preferred version is 410 or higher.
### Step 2 (Installing nVidia Drivers):
  - Open terminal and enter :
    ```shell_session
    $ sudo add-apt-repository ppa:graphics-drivers/ppa
    ```
    This will add the graphics drivers repository to ubuntu. you can browse for more information here: https://launchpad.net/~graphics-drivers/+archive/ubuntu/ppa
    
  - Next,
    ```shell_session
    $ sudo apt-get update
    $ ubuntu-drivers list
    ```
    This will list all the available drivers, and you can choose which one you would like to install.
    ```shell_session
    $ ubuntu-drivers list
    nvidia-driver-396
    nvidia-driver-418
    nvidia-driver-430
    nvidia-driver-410
    nvidia-driver-390
    nvidia-driver-415
    ```
  - I am choosing driver-410, to install the driver type,
    ```shell_session
    $ sudo apt install nvidia-driver-410
    ```
  - After installation completely successfully, reboot the PC,
    ```shell_session
    $ sudo reboot
    ```
  - Now type nvidia-smi to find out the installed drivers and the hardware, you can use this command (watch nvidia-smi) to monitor the Graphics card later.
    ```shell_session
    $ nvidia-smi
    +-----------------------------------------------------------------------------+
    | NVIDIA-SMI 410.104       Driver Version: 410.104       CUDA Version: 10.0   |
    |-------------------------------+----------------------+----------------------+
    | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
    | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
    |===============================+======================+======================|
    |   0  Tesla V100-PCIE...  On   | 00000000:17:00.0 Off |                    0 |
    | N/A   42C    P0    27W / 250W |      0MiB / 32480MiB |      0%      Default |
    +-------------------------------+----------------------+----------------------+
    ```
## Installing CUDA 10.0 (If it ain't broke, don't fix it)
  
  - We are using version 10.0 since it is the working CUDA driver with Tensorflow 2.0 without any bugs. You may upgrade in the future but for now, let's stick with it.
  
### Step 1 (Installation):
  - Go to nVidia cuda archive https://developer.nvidia.com/cuda-10.0-download-archive?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1804&target_type=debnetwork and download the installer type: Network
  - link to the installer : https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-repo-ubuntu1804_10.0.130-1_amd64.deb
  - Then use the following commands inside the folder where you have downloaded the installer,
  ```shell_session
  $ sudo dpkg -i cuda-repo-ubuntu1804_10.0.130-1_amd64.deb
  $ sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
  $ sudo apt-get update
  $ sudo apt-get install cuda-10-0
  ```
  - NOTE: Make sure you use "sudo apt-get install cuda-10-0" incase if you have used "sudo apt-get install cuda" , it will insatll cuda-10.1
 
### Step 2 (Adding PATH):
  - After the installation we have to set the path for the Graphics card to locate the CUDA libraries. open,
  ```shell_session
  $ sudo nano ~/.bashrc
  ```
  - Enter the following and save the bashrc to set the PATH variables.
  ```shell_session
  export PATH=/usr/local/cuda-10.0/bin${PATH:+:$PATH}}
  export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
  ```

## Installing cuDNN

  - Download the cuDNN from https://developer.nvidia.com/rdp/cudnn-archive
  - You should have a membership account to download the cuDNN libraries.
  - After logging in download cuDNN v7.6.0 for CUDA 10.0, Get the Linux tar package.
    https://developer.nvidia.com/compute/machine-learning/cudnn/secure/7.6.4.38/Production/10.0_20190923/cudnn-10.0-linux-x64-v7.6.4.38.tgz
  - Go to the folder where the archive got downloaded, in the terminal,
  ```shell_session
  $ tar -xzvf cudnn-10.0-linux-x64-v7.6.4.38.tgz
  ```
  - Then type the following to move the libraries to the appropriate folders,
  ```shell_session
  $ sudo cp cuda/include/cudnn.h /usr/local/cuda/include
  $ sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64
  $ sudo chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda/lib64/libcudnn*
  ```

## Installing pip3 and Tensorflow-2.0-GPU

  - The support for python v2.7 end official from 2020, So it's better if we can stick with python version 3.6

### Step 1 (Installing pip3):
  - Use the following commend to install pip3 in your PC,
  ```shell_session
  $ sudo apt-get install python3-pip
  $ sudo pip3 --upgrade pip
  ```
  
### Step 2 (Installing Tensorflow):
  - Now let's install Tensorflow 2.0
  ```shell_session
  $ pip3 install tensorflow-gpu==2.0.0
  ```
### Step 3 (Verifying the installation):
  - Run the following inside python3 terminal to verify the installation
  ```shell_session
  $ python3
  >>> import tensorflow as tf
  >>> hello = tf.constant('hello tensorflow')
  >>> x = [[2.]]
  >>> print('hello, {}'.format(tf.matmul(x, x)))
  2019-00-00 16:04:38.589080: I tensorflow/stream_executor/platform/default/dso_loader.cc:42] Successfully opened dynamic library   libcublas.so.10.0
  hello, [[4.]]
  >>> exit()
  ```
### That's it you have successfully Tensorflow-2.0-GPU with CUDA 10.0.
  

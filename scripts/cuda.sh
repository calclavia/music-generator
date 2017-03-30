# Tensorflow setup on Linux instance
# http://www.nvidia.com/object/gpu-accelerated-applications-tensorflow-installation.html
# https://www.tensorflow.org/install/install_linux
# Run this file from the parent dir of the repository

# Download CUDA
if [ ! -f "cuda.deb" ]
then
  wget -O cuda.deb https://developer.nvidia.com/compute/cuda/8.0/Prod2/local_installers/cuda-repo-ubuntu1404-8-0-local-ga2_8.0.61-1_amd64-deb
fi

# Ensure same name is used
sudo dpkg -i cuda.deb
yes | sudo apt-get update
yes | sudo apt-get install cuda

# Install libcupti-dev library
yes | sudo apt-get install libcupti-dev

# Set CUDA library paths
echo 'export LD_LIBRARY_PATH=/silo/cuda/lib64:/usr/local/cuda-8.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' >> ~/.bashrc
echo 'export PATH=/usr/local/cuda-8.0/bin${PATH:+:${PATH}}' >> ~/.bashrc
source ~/.bashrc

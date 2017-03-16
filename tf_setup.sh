# Tensorflow setup on Linux instance
# http://www.nvidia.com/object/gpu-accelerated-applications-tensorflow-installation.html

# Install CUDA
wget https://developer.nvidia.com/compute/cuda/8.0/Prod2/local_installers/cuda-repo-ubuntu1404-8-0-local-ga2_8.0.61-1_amd64-deb

# Ensure same name is used
sudo dpkg -i cuda-repo-ubuntu1404-8-0-local-ga2_8.0.61-1_amd64.deb
sudo apt-get update
sudo apt-get install cuda


sudo apt-get install python3 python3-pip python3-dev

# For Python midi
sudo apt-get install libasound2-dev python3-augeas swig

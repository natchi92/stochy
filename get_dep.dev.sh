#!/bin/sh
if ! [ -x "$(command -v sudo)" ]; then
  echo 'Warning: sudo is not installed.' >&2
  alias sudo=''
fi

sudo apt-get -y install cmake
sudo apt-get -y install libboost-all-dev
sudo apt-get -y install libopenblas-dev, 
sudo apt-get -y install liblapack-dev

#TODO: add the following as submodules

sudo apt-get -y install libarmadillo-dev
sudo apt-get -y install libmatio-dev
sudo apt-get -y install libginac-dev

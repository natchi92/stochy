#!/bin/sh
if ! [ -x "$(command -v sudo)" ]; then
  echo 'Warning: sudo is not installed.' >&2
  alias sudo=''
fi

sudo apt-get -y install cmake
sudo apt-get -y install libboost-all-dev
sudo apt-get -y install libopenblas-dev
sudo apt-get -y install liblapack-dev

#TODO: add the following as submodules

echo "--------------------------------"
echo "working on armadillo ..."
mkdir -p armadillo
cd armadillo
wget https://kent.dl.sourceforge.net/project/arma/armadillo-10.4.1.tar.xz
tar -xvf armadillo-10.4.1.tar.xz --strip-components 1
rm armadillo-10.4.1.tar.xz
./configure
make
sudo make install
echo "--------------------------------"
cd ../

sudo apt-get -y install libmatio-dev
sudo apt-get -y install libginac-dev

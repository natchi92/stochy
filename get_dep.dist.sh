#!/bin/sh
if ! [ -x "$(command -v sudo)" ]; then
  echo 'Warning: sudo is not installed.' >&2
  alias sudo=''
fi

sudo apt-get -y install libarmadillo9
sudo apt-get -y install libmatio4
sudo apt-get -y install libginac6

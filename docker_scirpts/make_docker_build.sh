#!/bin/sh
cd ../
git submodule update --init --recursive
docker build -t stochy -f Dockerfile.build .

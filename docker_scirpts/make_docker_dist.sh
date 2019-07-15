cd ../
if [ ! -d "build" ]; then
 echo ERROR: Please build stochy first
 exit 1 # terminate and indicate error
fi
cd build
rm -rf stochy-1.0.0.zip
rm -rf stochy-1.0.0
make package
unzip stochy-1.0.0.zip
cd ../
docker build -t stochy -f Dockerfile.dist .

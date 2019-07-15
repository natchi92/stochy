rm -rf build
cmake -H. -Bbuild -DCMAKE_BUILD_TYPE=Debug
cd build
make 
cd ../

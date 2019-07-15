rm -rf build
cmake -H. -Bbuild -DCMAKE_BUILD_TYPE=Release
cd build
make 
cd ../

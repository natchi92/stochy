rm -rf build
cmake -Wno-deprecated -Wimplicit-function-declaration -H. -Bbuild -DCMAKE_BUILD_TYPE=Release 
cd build
nice make -j20
cd ../

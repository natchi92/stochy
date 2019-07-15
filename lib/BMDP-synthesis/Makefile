#######################################################
# Specify the name of the executable and source files #
#######################################################
# This is the name of the executable to build
BIN  = synthesis
# Space separated list of source files to compile
SRCS = synthesis.cpp bmdp/BMDP.cpp bmdp/IntervalValueIteration.cpp

################################################################
# Information regarding OMPL location and the compiler options #
################################################################
# The installation prefix for OMPL (/usr/local for Linux,/opt/local for OS X)
#OMPL_DIR = ${HOME}/omplapp/build/Release
# The compiler options.  Change to -g when debugging code
CXX_FLAGS = -g -O2
# A list of directories to include during compilation
INCLUDE_FLAGS = -I.
# The linker options.  These include linking the ompl and ompl_app libraries for the executable
#LD_FLAGS = -L${OMPL_DIR}/lib -lompl -lompl_app -lboost_serialization -ltriangle -Wl,-rpath ${OMPL_DIR}/lib
# The c++ compiler to invoke, if not already set
#CXX=c++

################################################
# Nothing below this should need to be changed #
################################################
OBJS=$(SRCS:.cpp=.o)

$(BIN): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $(BIN) $(OBJS) $(LTLOBJS) $(LD_FLAGS)

lib:
	set -x
	g++ -Wall -fPIC -c -g -O2 -I. synthesis.cpp  -o synthesis.o
	g++ -Wall -fPIC -c -g -O2 -I. bmdp/BMDP.cpp  -o bmdp/BMDP.o
	g++ -Wall -fPIC -c -g -O2 -I. bmdp/IntervalValueIteration.cpp  -o bmdp/IntervalValueIteration.o
	mkdir -p lib
	ar rcs lib/libsynth.a synthesis.o bmdp/BMDP.o bmdp/IntervalValueIteration.o

clean:
	rm *.o
	rm $(BIN)
	rm bmdp/*.o

%.o: %.cpp
	$(CXX) -c $(CXX_FLAGS) $(INCLUDE_FLAGS) $< -o $@

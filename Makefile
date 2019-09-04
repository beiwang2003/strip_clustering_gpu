CXXFLAGS += -std=c++14 -g -O3 -fopenmp
LDFLAGS += -std=c++14 -fopenmp
CC = c++

strip-cluster : strip-cluster.o 
strip-cluster.o: strip-cluster.cc

CXXFLAGS += -std=c++14 -g #-O3
LDFLAGS += -std=c++14
CC = c++

strip-cluster : strip-cluster.o 
strip-cluster.o: strip-cluster.cc

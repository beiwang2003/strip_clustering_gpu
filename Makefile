#CXXFLAGS += -std=c++14 -g -O3 -fopenmp -march=native
#LDFLAGS += -std=c++14 -fopenmp -march=native
#CC = c++
CXXFLAGS += --c++14 -g -O3 -acc -Minfo=accel -ta=nvidia:managed -Mnodepchk
LDFLAGS += --c++14 -acc -Minfo=accel -ta=nvidia:managed -Mnodepchk
CC = pgc++

strip-cluster : strip-cluster.o 
strip-cluster.o: strip-cluster.cc

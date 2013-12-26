NVCC = nvcc
CCFLAGS = -O3
NVCCFLAGS = 
LDFLAGS = 
SRCS = $(wildcard *.cu)
OBJS = $(patsubst %.cu, %.o, $(SRCS))
all:$(OBJS)
%.o: %.cu
	$(NVCC) -o $@ $< $(CCFLAGS) $(NVCCFLAGS) $(LDFLAGS)
clean:
	rm *.o

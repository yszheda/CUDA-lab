CC = clang
CCFLAGS = -O3
LDFLAGS = -lrt
SRCS = $(wildcard *.c)
OBJS = $(patsubst %.c, %.o, $(SRCS))
all:$(OBJS)
%.o: %.c
	$(CC) -o $@ $< $(CCFLAGS) $(LDFLAGS)
clean:
	rm *.o

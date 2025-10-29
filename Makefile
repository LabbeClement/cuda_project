CC = nvcc
TARGET = mlp
SRCS = src/mlp.cu
OBJS = $(SRCS:.cu=.o)
.PHONY: all clean
all: $(TARGET)
$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^
%.o: %.cu
	$(CC) $(CFLAGS) -c $< -o $@
clean:
	rm -f $(OBJS) $(TARGET)
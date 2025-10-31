CC := nvcc
TARGET := mlp
SRCS := src/mlp.cu
OBJS := $(SRCS:.cu=.o)

# find test sources and create binaries under tests/bin
TEST_SRCS := $(shell find tests -name '*.cu')
TEST_BINS := $(patsubst tests/%.cu,tests/bin/%,$(TEST_SRCS))

.PHONY: all clean tests run-tests
all: $(TARGET) tests

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^

%.o: %.cu
	$(CC) $(CFLAGS) -c $< -o $@

tests: $(TEST_BINS)

# compile each test source into tests/bin/<path>
tests/bin/%: tests/%.cu
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -o $@ $<

run-tests: tests
	@echo "Running all tests..."
	@for t in $(TEST_BINS); do \
		echo "---- $$t ----"; \
		./$$t || { echo "Test failed: $$t"; exit 1; }; \
	done

clean:
	rm -f $(OBJS) $(TARGET)
	rm -rf tests/bin
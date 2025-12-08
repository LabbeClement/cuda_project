CC := nvcc
CFLAGS := -arch=sm_75 -O3 -Xcompiler -Wall -lcublas -std=c++14

# 1. LIBRAIRIE PRINCIPALE
LIB_SRCS := src/mlp.cu src/cnn.cu
LIB_OBJS := $(LIB_SRCS:.cu=.o)

# 2. FICHIERS DE TEST
TEST_SRCS := $(shell find tests -name '*.cu')
TEST_BINS := $(patsubst tests/%.cu,tests/bin/%,$(TEST_SRCS))

# Séparer les tests fonctionnels des benchmarks
BENCH_TESTS := $(filter %_benchmark, $(TEST_BINS))
FUNCTIONAL_TESTS := $(filter-out %_benchmark, $(TEST_BINS))

# Séparer les benchmarks MLP et CNN
BENCH_TESTS_MLP := $(filter tests/bin/mlp/%, $(BENCH_TESTS))
BENCH_TESTS_CNN := $(filter tests/bin/cnn/%, $(BENCH_TESTS))

# Fichiers de benchmark Python
PYTHON_BENCH_MLP := tests/pytorch_benchmark/pytorch_benchmark.py
PYTHON_BENCH_CNN := tests/pytorch_benchmark/cnn_benchmark.py

.PHONY: all clean tests run-tests run-benchmarks setup-bench-dir graph

all: $(LIB_OBJS) tests

# --- REGLES DE COMPILATION ---

%.o: %.cu
	$(CC) $(CFLAGS) -c $< -o $@

tests/bin/%: tests/%.cu $(LIB_OBJS)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -o $@ $< $(LIB_OBJS)

tests: $(TEST_BINS) setup-bench-dir

setup-bench-dir:
	@mkdir -p tests/benchmark
	@mkdir -p tests/data

# --- EXECUTION ---

run-tests: $(FUNCTIONAL_TESTS)
	@echo "Running functional tests..."
	@for t in $(FUNCTIONAL_TESTS); do \
		echo "---- $$t ----"; \
		./$$t || { echo "Test failed: $$t"; exit 1; }; \
	done

run-benchmarks:
	@echo "========================================================"
	@echo "===        PART 1: MLP BENCHMARKS (CUDA vs PyTorch)  ==="
	@echo "========================================================"
	
	@echo "\n--- CUDA MLP BENCHMARKS ---"
	@for t in $(BENCH_TESTS_MLP); do \
		echo "\n>>> Executing $$t ..."; \
		./$$t || { echo "CUDA Benchmark failed: $$t"; exit 1; }; \
	done

	@echo "\n--- PYTORCH MLP REFERENCE ---"
	@python3 $(PYTHON_BENCH_MLP)

	@echo "\n\n========================================================"
	@echo "===        PART 2: CNN BENCHMARKS (CUDA vs PyTorch)  ==="
	@echo "========================================================"

	@echo "\n--- CUDA CNN BENCHMARKS ---"
	@for t in $(BENCH_TESTS_CNN); do \
		echo "\n>>> Executing $$t ..."; \
		./$$t || { echo "CUDA Benchmark failed: $$t"; exit 1; }; \
	done

	@echo "\n--- PYTORCH CNN REFERENCE ---"
	@python3 $(PYTHON_BENCH_CNN)

	@echo "\n========================================================"
	@echo "=== BENCHMARK COMPLETE ==="


graph:
	@python3 $(PYTHON_PLOT)

clean:
	rm -f $(LIB_OBJS)
	rm -rf tests/bin
	rm -rf plots
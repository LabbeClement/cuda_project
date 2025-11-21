CC := nvcc
CFLAGS := -arch=sm_35 -O3 -Xcompiler -Wall # Ajout de flags d'optimisation et d'avertissement

# 1. LIBRAIRIE PRINCIPALE
LIB_SRCS := src/mlp.cu
LIB_OBJS := $(LIB_SRCS:.cu=.o) # src/mlp.o

# 2. FICHIERS DE TEST
TEST_SRCS := $(shell find tests/mlp -name '*.cu')
TEST_BINS := $(patsubst tests/%.cu,tests/bin/%,$(TEST_SRCS))

# Séparer les tests fonctionnels des benchmarks
BENCH_TESTS := $(filter %_benchmark, $(TEST_BINS))
FUNCTIONAL_TESTS := $(filter-out %_benchmark, $(TEST_BINS))

# Fichier de benchmark Python
PYTHON_BENCH := tests/pytorch_benchmark/pytorch_benchmark.py

.PHONY: all clean tests run-tests run-benchmarks setup-bench-dir
all: $(LIB_OBJS) tests

# --- REGLES DE COMPILATION ---

# Règle pour compiler les objets (.cu -> .o)
# Indentation TAB OBLIGATOIRE
%.o: %.cu
	$(CC) $(CFLAGS) -c $< -o $@

# Règle de linkage pour les exécutables de test
# Elle lie chaque test source ($<) avec l'objet de la librairie (src/mlp.o)
tests/bin/%: tests/%.cu $(LIB_OBJS)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -o $@ $< $(LIB_OBJS)

# Construction de la cible 'tests' (compile tout)
tests: $(TEST_BINS) setup-bench-dir

setup-bench-dir:
	@mkdir -p tests/benchmark

# --- EXECUTION ---

# Lancement des tests fonctionnels
run-tests: $(FUNCTIONAL_TESTS)
	@echo "Running functional tests..."
	@for t in $(FUNCTIONAL_TESTS); do \
		echo "---- $$t ----"; \
		./$$t || { echo "Test failed: $$t"; exit 1; }; \
	done

# Lancement des benchmarks de performance (Comparaison CUDA vs PyTorch)
run-benchmarks: $(BENCH_TESTS)
	@echo "========================================================"
	@echo "=== STARTING BENCHMARK: CUDA vs PYTORCH REFERENCE ===="
	@echo "========================================================"

	@echo "\n--- RUNNING CUSTOM CUDA BENCHMARK (Naïf) ---"
	@for t in $(BENCH_TESTS); do \
		./$$t || { echo "CUDA Benchmark failed: $$t"; exit 1; }; \
	done

	@echo "\n--- RUNNING PYTORCH REFERENCE BENCHMARK ---"
	@python3 $(PYTHON_BENCH)

	@echo "========================================================"
	@echo "=== BENCHMARK COMPLETE. COMPARE THE TIMES ABOVE. ======="
	@echo "========================================================"

# --- NETTOYAGE ---

clean:
	rm -f $(LIB_OBJS)
	rm -rf tests/bin tests/pytorch_benchmark
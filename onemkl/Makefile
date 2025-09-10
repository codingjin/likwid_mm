.PHONY: all clean perf sgemm

# Default target: build everything
all: perf sgemm

# Build openblas_sgemm_perf
perf:
	$(MAKE) -f Makefile_perf

# Build openblas_sgemm
sgemm:
	$(MAKE) -f Makefile_sgemm

# Clean everything
clean:
	$(MAKE) -f Makefile_perf clean
	$(MAKE) -f Makefile_sgemm clean
#!/bin/bash

# Check if exactly 4 arguments are provided
if [ "$#" -ne 4 ]; then
    echo "Usage: $0 M K N threadnum"
    echo "Example: $0 4096 4096 4096 16"
    exit 1
fi

./settings.sh

# Assign arguments to variables
M=$1
K=$2
N=$3
THREADS=$4

make

# Run the program, display output and append to perf.report
./onemkl_sgemm_perf "$M" "$K" "$N" "$THREADS" 2>&1 | tee perf.report
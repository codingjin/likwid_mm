#!/bin/bash
#set -x
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

# Remove previous report
REPORT="likwid-powermeter.report"
if [ -f "$REPORT" ]; then
    cp "$REPORT" "${REPORT}.bak"
fi
rm -f "$REPORT"

make
# Warmup rounds
echo "Warming up ..."
for ((i=0; i<2; i++)); do
    likwid-powermeter ./openblas_sgemm "$M" "$K" "$N" "$THREADS" >/dev/null
done

# Main measurement
echo "Start ..."
for ((i=0; i<20; i++)); do
    likwid-powermeter ./openblas_sgemm "$M" "$K" "$N" "$THREADS" 2>&1 | tee -a "$REPORT"
done

echo "Done"

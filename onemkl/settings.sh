#!/bin/bash

# sudo apt update; sudo apt install linux-tools-$(uname -r)
# disable CPU scaling
sudo cpupower frequency-set -g performance

# lock to max frequency
MAX=$(cat /sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq)
for cpu in /sys/devices/system/cpu/cpu*/cpufreq/scaling_{min,max}_freq; do
  echo $MAX | sudo tee $cpu
done

# disable turbo
echo 1 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo

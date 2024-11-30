#!/bin/bash

make clean && make

output_file="result.txt"
filename="img/sample_5184×3456.bmp"
runs=5

declare -a mem_times=()
declare -a kernel_times=()
declare -a serial_times=()
declare -a cuda_times=()
declare -a speedups=()


echo -e "\nRunning tests..."
for i in $(seq 1 $runs); do
   printf "Test %d/%d [" $i $runs
   # Progress bar
   for j in $(seq 1 20); do
       if [ $j -le $(( i * 20 / runs )) ]; then
           printf "#"
       else
           printf " "
       fi
   done
   printf "] %d%%\r" $(( i * 100 / runs ))
   
   output=$(./jpeg "$filename" output_img/41.jpg)
   
   mem_time=$(echo "$output" | grep "Memory allocation" | sed 's/.*time: \([0-9.]*\).*/\1/')
   kernel_time=$(echo "$output" | grep "Kernel Launching" | sed 's/.*Time : \([0-9.]*\).*/\1/')
   serial_time=$(echo "$output" | grep "Serial Version" | sed 's/.*: \([0-9.]*\).*/\1/')
   cuda_time=$(echo "$output" | grep "CUDA Version" | sed 's/.*: \([0-9.]*\).*/\1/')
   speedup=$(echo "$output" | grep "Speedup" | sed 's/.*: \([0-9.]*\).*/\1/')
   
   mem_times+=($mem_time)
   kernel_times+=($kernel_time)
   serial_times+=($serial_time)
   cuda_times+=($cuda_time)
   speedups+=($speedup)
done
echo -e "\n\nTests completed."

mem_avg=$(awk 'BEGIN {sum=0} {sum+=$1} END {printf "%.2f", sum/NR}' <<< "${mem_times[*]}")
kernel_avg=$(awk 'BEGIN {sum=0} {sum+=$1} END {printf "%.2f", sum/NR}' <<< "${kernel_times[*]}")
serial_avg=$(awk 'BEGIN {sum=0} {sum+=$1} END {printf "%.3f", sum/NR}' <<< "${serial_times[*]}")
cuda_avg=$(awk 'BEGIN {sum=0} {sum+=$1} END {printf "%.3f", sum/NR}' <<< "${cuda_times[*]}")
speedup_avg=$(awk 'BEGIN {sum=0} {sum+=$1} END {printf "%.3f", sum/NR}' <<< "${speedups[*]}")

cat > "$output_file" << EOF
================================= Performance Results ==========================
Average over $runs runs:

CUDA Memory Operations:
   Memory Allocation & Transfer : $mem_avg ms
   Kernel Execution             : $kernel_avg ms

Execution Time:
   Serial Version               : $serial_avg seconds
   CUDA Version                 : $cuda_avg seconds
   Speedup                      : ${speedup_avg}x

================================================================================
EOF
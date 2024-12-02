#!/bin/bash

make clean && make

output_file="result.txt"
filename="img/sample_5184×3456.bmp"
runs=10
use_srun=0

# 解析命令行參數
for arg in "$@"; do
    case $arg in
        -s=*|--srun=*)
        use_srun="${arg#*=}"
        shift
        ;;
    esac
done

declare -a host_to_dev_times=()
declare -a kernel_times=()
declare -a dev_to_host_times=()
declare -a serial_times=()
declare -a cuda_times=()
declare -a speedups=()

# 定義執行命令
if [ "$use_srun" = "1" ]; then
    run_cmd="srun ./jpeg"
else
    run_cmd="./jpeg"
fi

echo -e "\nRunning tests with${use_srun:+" srun"} command..."
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

   output=$($run_cmd "$filename" output_img/41.jpg)

   host_to_dev_time=$(echo "$output" | grep "Memory transfer from \[Host\] to \[Device\]" | sed 's/.*time: \([0-9.]*\).*/\1/')
   kernel_time=$(echo "$output" | grep "Kernel Launching Time" | sed 's/.*Time : \([0-9.]*\).*/\1/')
   dev_to_host_time=$(echo "$output" | grep "Memory transfer from \[Device\] to \[Host\]" | sed 's/.*time: \([0-9.]*\).*/\1/')
   serial_time=$(echo "$output" | grep "Serial Version" | sed 's/.*: \([0-9.]*\).*/\1/')
   cuda_time=$(echo "$output" | grep "CUDA Version" | sed 's/.*: \([0-9.]*\).*/\1/')
   speedup=$(echo "$output" | grep "Speedup" | sed 's/.*: \([0-9.]*\).*/\1/')

   host_to_dev_times+=($host_to_dev_time)
   kernel_times+=($kernel_time)
   dev_to_host_times+=($dev_to_host_time)
   serial_times+=($serial_time)
   cuda_times+=($cuda_time)
   speedups+=($speedup)
done
echo -e "\n\nTests completed."

host_to_dev_avg=$(awk 'BEGIN {sum=0} {sum+=$1} END {printf "%.2f", sum/NR}' <<< "${host_to_dev_times[*]}")
kernel_avg=$(awk 'BEGIN {sum=0} {sum+=$1} END {printf "%.2f", sum/NR}' <<< "${kernel_times[*]}")
dev_to_host_avg=$(awk 'BEGIN {sum=0} {sum+=$1} END {printf "%.2f", sum/NR}' <<< "${dev_to_host_times[*]}")
serial_avg=$(awk 'BEGIN {sum=0} {sum+=$1} END {printf "%.3f", sum/NR}' <<< "${serial_times[*]}")
cuda_avg=$(awk 'BEGIN {sum=0} {sum+=$1} END {printf "%.3f", sum/NR}' <<< "${cuda_times[*]}")
speedup_avg=$(awk 'BEGIN {sum=0} {sum+=$1} END {printf "%.3f", sum/NR}' <<< "${speedups[*]}")

cat > "$output_file" << EOF
================================= Performance Results ==========================
Average over $runs runs:

CUDA Memory Operations:
   Host to Device Transfer      : $host_to_dev_avg ms
   Kernel Execution             : $kernel_avg ms
   Device to Host Transfer      : $dev_to_host_avg ms

Execution Time:
   Serial Version               : $serial_avg seconds
   CUDA Version                 : $cuda_avg seconds
   Speedup                      : ${speedup_avg}x

================================================================================
EOF
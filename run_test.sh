#!/bin/bash

make clean && make

output_file="result.txt"
default_image="img/sample3.bmp"
runs=10
use_srun=0
run_all=0

# 解析參數
for arg in "$@"
do
    case $arg in
        -s)
        use_srun=1
        ;;
        -a|--all)
        run_all=1
        ;;
    esac
done

# 定義執行命令
if [ "$use_srun" = "1" ]; then
    run_cmd="srun ./jpeg"
else
    run_cmd="./jpeg"
fi

# 清空或創建結果文件
echo "================================= Performance Results ==========================" > "$output_file"

if [ "$run_all" = "1" ]; then
    # 運行所有測試圖片
    for img in test_images/*.bmp; do
        img_name=$(basename "$img")
        echo -e "\nTesting image: $img_name"
        echo -e "\nResults for image: $img_name" >> "$output_file"
        
        declare -a host_to_dev_times=()
        declare -a kernel_times=()
        declare -a dev_to_host_times=()
        declare -a serial_times=()
        declare -a cuda_times=()
        declare -a speedups=()

        echo "Running tests..."
        for i in $(seq 1 $runs); do
            printf "Test %d/%d [" $i $runs
            for j in $(seq 1 20); do
                if [ $j -le $(( i * 20 / runs )) ]; then
                    printf "#"
                else
                    printf " "
                fi
            done
            printf "] %d%%\r" $(( i * 100 / runs ))

            output=$($run_cmd "$img" "output_img/${img_name%.bmp}.jpg")

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
        echo -e "\nTests completed for $img_name"

        # 計算平均值
        host_to_dev_avg=$(awk 'BEGIN {sum=0} {sum+=$1} END {printf "%.2f", sum/NR}' <<< "${host_to_dev_times[*]}")
        kernel_avg=$(awk 'BEGIN {sum=0} {sum+=$1} END {printf "%.2f", sum/NR}' <<< "${kernel_times[*]}")
        dev_to_host_avg=$(awk 'BEGIN {sum=0} {sum+=$1} END {printf "%.2f", sum/NR}' <<< "${dev_to_host_times[*]}")
        serial_avg=$(awk 'BEGIN {sum=0} {sum+=$1} END {printf "%.3f", sum/NR}' <<< "${serial_times[*]}")
        cuda_avg=$(awk 'BEGIN {sum=0} {sum+=$1} END {printf "%.3f", sum/NR}' <<< "${cuda_times[*]}")
        speedup_avg=$(awk 'BEGIN {sum=0} {sum+=$1} END {printf "%.3f", sum/NR}' <<< "${speedups[*]}")

        # 寫入結果
        cat >> "$output_file" << EOF

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
    done
else
    # 運行單一預設圖片
    echo -e "\nTesting default image: $default_image"
    echo -e "\nResults for default image: $(basename $default_image)" >> "$output_file"
    
    declare -a host_to_dev_times=()
    declare -a kernel_times=()
    declare -a dev_to_host_times=()
    declare -a serial_times=()
    declare -a cuda_times=()
    declare -a speedups=()

    # ... [原本的單一圖片測試代碼] ...
    # 這裡是原本的測試邏輯，保持不變
    echo "Running tests..."
    for i in $(seq 1 $runs); do
        printf "Test %d/%d [" $i $runs
        for j in $(seq 1 20); do
            if [ $j -le $(( i * 20 / runs )) ]; then
                printf "#"
            else
                printf " "
            fi
        done
        printf "] %d%%\r" $(( i * 100 / runs ))

        output=$($run_cmd "$default_image" output_img/default.jpg)

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
    echo -e "\nTests completed."

    host_to_dev_avg=$(awk 'BEGIN {sum=0} {sum+=$1} END {printf "%.2f", sum/NR}' <<< "${host_to_dev_times[*]}")
    kernel_avg=$(awk 'BEGIN {sum=0} {sum+=$1} END {printf "%.2f", sum/NR}' <<< "${kernel_times[*]}")
    dev_to_host_avg=$(awk 'BEGIN {sum=0} {sum+=$1} END {printf "%.2f", sum/NR}' <<< "${dev_to_host_times[*]}")
    serial_avg=$(awk 'BEGIN {sum=0} {sum+=$1} END {printf "%.3f", sum/NR}' <<< "${serial_times[*]}")
    cuda_avg=$(awk 'BEGIN {sum=0} {sum+=$1} END {printf "%.3f", sum/NR}' <<< "${cuda_times[*]}")
    speedup_avg=$(awk 'BEGIN {sum=0} {sum+=$1} END {printf "%.3f", sum/NR}' <<< "${speedups[*]}")

    cat >> "$output_file" << EOF

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
fi
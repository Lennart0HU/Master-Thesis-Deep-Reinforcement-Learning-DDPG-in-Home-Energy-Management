#!/bin/bash
gpu_idx=0
SECONDS=0 # Reset the SECONDS variable at the start of the script
start_time=$SECONDS
MIN_MEMORY=25000
MIN_SYS_MEMORY=80000000 # Minimum system memory in KB (adjust as needed)

# Function to check available GPU memory
check_gpu_memory() {
    while true; do
        # Get the available memory on both GPUs
        available_memory=($(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits))
        available_memory_0=${available_memory[0]}
        available_memory_1=${available_memory[1]}
        
        # Get the GPU utilization on both GPUs
        gpu_util=($(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits))
        gpu_util_0=${gpu_util[0]}
        gpu_util_1=${gpu_util[1]}
        
        # Check if both GPUs have less than the minimum required memory
        if [ "$available_memory_0" -lt "$MIN_MEMORY" ] && [ "$available_memory_1" -lt "$MIN_MEMORY" ]; then
            echo "Both GPUs have less than $MIN_MEMORY memory available. Waiting for 5 minutes..."
            sleep 300
            continue
        fi
        
        # Check if both GPUs are fully utilized
        if [ "$gpu_util_0" -gt 95 ] && [ "$gpu_util_1" -gt 95 ]; then
            echo "Both GPUs are fully utilized. Waiting for 2 minutes..."
            sleep 90
            continue
        fi
        
        # Choose the GPU with more available capacity (considering both memory and utilization)
        if [ "$available_memory_0" -ge "$available_memory_1" ] && [ "$gpu_util_0" -lt 95 ]; then
            GPU_ID=0
        elif [ "$gpu_util_1" -lt 95 ]; then
            GPU_ID=1
        else
            GPU_ID=0
        fi
        
        break
    done
}

# Function to check available system memory
check_system_memory() {

    total_available_sys_memory=($(awk '/MemAvailable/ {print $2}' /proc/meminfo) - $(awk '/Buffers/ {print $2}' /proc/meminfo) - $(awk '/Cached/ {print $2}' /proc/meminfo))
    
    if [ "$total_available_sys_memory" -lt "$MIN_SYS_MEMORY" ]; then
        echo "System memory ($total_available_sys_memory) is below $MIN_SYS_MEMORY KB. Waiting for 5 minutes..."
        sleep 300
        check_system_memory
    fi
}



#JOB_ID=10219800
JOB_IDS=(
    10769802
    10769805
    10769808
    10769811
    10769814
    10769823
    10769832
    10769841
    10769844
    10769850
)

#while ((JOB_ID <= 10219800))
#do
for JOB_ID in "${JOB_IDS[@]}"; do
    export JOB_ID
    cp input06_GS_eval_40.jl out/input/$JOB_ID--input.jl

    for (( TASK_ID=11; TASK_ID<=40; TASK_ID++ ))  
    do  
        # Check GPU memory and system memory
        check_system_memory
        check_gpu_memory
        echo "Available memories: GPU0: $available_memory_0, Util: $gpu_util_0. GPU1: $available_memory_1, Util: $gpu_util_1. System: $total_available_sys_memory KB"
        TASK_ID=$TASK_ID GPU_ID=$GPU_ID julia DDPG_reinforce_charger_v1_40.jl &
        sleep 90
    done
    
    sleep 120

done

wait


end_time=$SECONDS
elapsed_minutes=$(( (end_time - start_time) / 60 ))
echo "ALL JOBS FINISHED. TIME ELAPSED: $elapsed_minutes minutes"

wait

# NECCESARY SETTINGS BEFORE RUNNING:
    # New JOB_ID
    # Set TASK_ID
    # Check num_seeds == highest TASK_ID
    # Set NUM_EP
    # Adjust WAIT suitable for NUM_EP and number of GPUs used.
    # Change Job_Id or Task_Id dependend varialbes if needed (such as discomfort-weight)
    # Save all

    # for rule-based:
    # TASK_ID = 1:1
    # num_seeds = 1
    # adjust track to <0
    # train, plot, plot_all = 0
    # case: "(Charger_ID)_$(season)_$(price)_rule_based_$(track)"
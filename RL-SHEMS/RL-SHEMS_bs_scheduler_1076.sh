#!/bin/bash
gpu_idx=0
SECONDS=0 # Reset the SECONDS variable at the start of the script
start_time=$SECONDS
MIN_MEMORY=30000
MIN_SYS_MEMORY=60000000 # Minimum system memory in KB (adjust as needed)

# Function to check available GPU memory
check_gpu_memory() {
    # Get the available memory on both GPUs
    available_memory=($(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits))
    available_memory_0=${available_memory[0]}
    available_memory_1=${available_memory[1]}
    
    # Check if both GPUs have less than the minimum required memory
    if [ "$available_memory_0" -lt "$MIN_MEMORY" ] && [ "$available_memory_1" -lt "$MIN_MEMORY" ]; then
        echo "Both GPUs have less than $MIN_MEMORY memory available. Waiting for 2 minutes..."
        sleep 120
        check_gpu_memory
    fi
    
    # Choose the GPU with more available memory
    if [ "$available_memory_0" -ge "$available_memory_1" ]; then
        GPU_ID=0
    else
        GPU_ID=1
    fi
}

# Function to check available system memory
check_system_memory() {

    total_available_sys_memory=($(awk '/MemAvailable/ {print $2}' /proc/meminfo) - $(awk '/Buffers/ {print $2}' /proc/meminfo) - $(awk '/Cached/ {print $2}' /proc/meminfo))
    
    if [ "$total_available_sys_memory" -lt "$MIN_SYS_MEMORY" ]; then
        echo "System memory ($total_available_sys_memory) is below $MIN_SYS_MEMORY KB. Waiting for 2 minutes..."
        sleep 120
        check_system_memory
    fi
}



JOB_ID=10769802

while ((JOB_ID <= 10769880))
do

    if [[ $JOB_ID -eq 10769803 || $JOB_ID -eq 10769809 || $JOB_ID -eq 10769827 || $JOB_ID -eq 10769854 ]]; then
        JOB_ID=$((JOB_ID + 1))
        continue
    fi

    export JOB_ID
    cp input06_GS_eval.jl out/input/$JOB_ID--input.jl
    
    # Check GPU memory and system memory
    check_system_memory
    check_gpu_memory
    echo "Available memories: GPU0: $available_memory_0, GPU1: $available_memory_1, System: $total_available_sys_memory KB"

    for (( TASK_ID=1; TASK_ID<=10; TASK_ID++ ))  
    do  
        check_system_memory
        TASK_ID=$TASK_ID GPU_ID=$GPU_ID julia DDPG_reinforce_charger_v1.jl &
        sleep 15
    done
    
    sleep 240

    JOB_ID=$((JOB_ID + 1))
    end_time=$SECONDS
    elapsed_minutes=$(( (end_time - start_time) / 60 ))
    echo "TIME ELAPSED: $elapsed_minutes minutes"
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
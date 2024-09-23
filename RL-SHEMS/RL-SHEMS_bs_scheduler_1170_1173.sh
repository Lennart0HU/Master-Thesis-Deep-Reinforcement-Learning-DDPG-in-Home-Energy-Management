#!/bin/bash
gpu_idx=0
SECONDS=0 # Reset the SECONDS variable at the start of the script
start_time=$SECONDS
MIN_MEMORY=20000
MIN_SYS_MEMORY=35000000 # Minimum system memory in KB (adjust as needed)

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

JOB_ID=11700100

while ((JOB_ID <= 11700900))
do
    export JOB_ID
    cp input00_RB_eval.jl out/input/$JOB_ID--input.jl
    
    # Check GPU memory and system memory
    check_system_memory
    check_gpu_memory
    echo "Available memories: GPU0: $available_memory_0, Util: $gpu_util_0. GPU1: $available_memory_1, Util: $gpu_util_1. System: $total_available_sys_memory KB"
    for (( TASK_ID=1; TASK_ID<=1; TASK_ID++ ))  
    do
        TASK_ID=$TASK_ID GPU_ID=$GPU_ID julia DDPG_reinforce_charger_v1.jl &
    done
    
    sleep 30

    JOB_ID=$((JOB_ID + 100))
    end_time=$SECONDS
    elapsed_minutes=$(( (end_time - start_time) / 60 ))
    echo "TIME ELAPSED: $elapsed_minutes minutes"
done

check_system_memory
check_gpu_memory

JOB_ID=11709800
export JOB_ID
cp input00_RB_eval.jl out/input/$JOB_ID--input.jl
for (( TASK_ID=1; TASK_ID<=1; TASK_ID++ ))  
do
    TASK_ID=$TASK_ID GPU_ID=$GPU_ID julia DDPG_reinforce_charger_v1.jl &
done

wait


JOB_ID=11710100

while ((JOB_ID <= 11710900))
do
    export JOB_ID
    cp input01_RB_test.jl out/input/$JOB_ID--input.jl
    
    # Check GPU memory and system memory
    check_system_memory
    check_gpu_memory
    echo "Available memories: GPU0: $available_memory_0, Util: $gpu_util_0. GPU1: $available_memory_1, Util: $gpu_util_1. System: $total_available_sys_memory KB"
    for (( TASK_ID=1; TASK_ID<=1; TASK_ID++ ))  
    do
        TASK_ID=$TASK_ID GPU_ID=$GPU_ID julia DDPG_reinforce_charger_v1.jl &
    done
    
    sleep 30

    JOB_ID=$((JOB_ID + 100))
    end_time=$SECONDS
    elapsed_minutes=$(( (end_time - start_time) / 60 ))
    echo "TIME ELAPSED: $elapsed_minutes minutes"
done

JOB_ID=11719800
export JOB_ID
cp input01_RB_test.jl out/input/$JOB_ID--input.jl
for (( TASK_ID=1; TASK_ID<=1; TASK_ID++ ))  
do
    TASK_ID=$TASK_ID GPU_ID=$GPU_ID julia DDPG_reinforce_charger_v1.jl &
done

wait



JOB_ID=11720100

while ((JOB_ID <= 11720900))
do
    export JOB_ID
    cp input02_DF_eval.jl out/input/$JOB_ID--input.jl
    
    # Check GPU memory and system memory

    for (( TASK_ID=1; TASK_ID<=40; TASK_ID++ ))  
    do
        check_system_memory
        check_gpu_memory
        echo "Available memories: GPU0: $available_memory_0, Util: $gpu_util_0. GPU1: $available_memory_1, Util: $gpu_util_1. System: $total_available_sys_memory KB"
        TASK_ID=$TASK_ID GPU_ID=$GPU_ID julia DDPG_reinforce_charger_v1.jl &
        sleep 75
    done
    
    sleep 300
    JOB_ID=$((JOB_ID + 100))
    end_time=$SECONDS
    elapsed_minutes=$(( (end_time - start_time) / 60 ))
    echo "TIME ELAPSED: $elapsed_minutes minutes"
done


JOB_ID=11729800
export JOB_ID
cp input02_DF_eval.jl out/input/$JOB_ID--input.jl
for (( TASK_ID=1; TASK_ID<=40; TASK_ID++ )) 
do
    check_system_memory
    check_gpu_memory
    echo "Available memories: GPU0: $available_memory_0, Util: $gpu_util_0. GPU1: $available_memory_1, Util: $gpu_util_1. System: $total_available_sys_memory KB"
    TASK_ID=$TASK_ID GPU_ID=$GPU_ID julia DDPG_reinforce_charger_v1.jl &
    sleep 90
done

wait


JOB_ID=11730100

while ((JOB_ID <= 11730900))
do
    export JOB_ID
    cp input03_DF_test.jl out/input/$JOB_ID--input.jl
    
    # Check GPU memory and system memory

    for (( TASK_ID=40; TASK_ID<=40; TASK_ID++ ))  
    do
        check_system_memory
        check_gpu_memory
        echo "Available memories: GPU0: $available_memory_0, Util: $gpu_util_0. GPU1: $available_memory_1, Util: $gpu_util_1. System: $total_available_sys_memory KB"
        TASK_ID=$TASK_ID GPU_ID=$GPU_ID julia DDPG_reinforce_charger_v1.jl &
        wait
    done
    
    wait
    JOB_ID=$((JOB_ID + 100))
    end_time=$SECONDS
    elapsed_minutes=$(( (end_time - start_time) / 60 ))
    echo "TIME ELAPSED: $elapsed_minutes minutes"
done

JOB_ID=11739800
export JOB_ID
cp input03_DF_test.jl out/input/$JOB_ID--input.jl
for (( TASK_ID=40; TASK_ID<=40; TASK_ID++ )) 
do
    check_system_memory
    check_gpu_memory
    echo "Available memories: GPU0: $available_memory_0, Util: $gpu_util_0. GPU1: $available_memory_1, Util: $gpu_util_1. System: $total_available_sys_memory KB"
    TASK_ID=$TASK_ID GPU_ID=$GPU_ID julia DDPG_reinforce_charger_v1.jl &
    wait
done

wait




end_time=$SECONDS
elapsed_minutes=$(( (end_time - start_time) / 60 ))
echo "ALL JOBS FINISHED. TIME ELAPSED: $elapsed_minutes minutes"

wait
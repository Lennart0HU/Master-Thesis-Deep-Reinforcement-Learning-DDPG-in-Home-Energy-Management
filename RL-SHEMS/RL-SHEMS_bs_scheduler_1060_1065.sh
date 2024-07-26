#!/bin/bash
gpu_idx=0
SECONDS=0 # Reset the SECONDS variable at the start of the script
start_time=$SECONDS
MIN_MEMORY=30000
MIN_SYS_MEMORY=80000000 # Minimum system memory in KB (adjust as needed)

# Function to check available GPU memory
check_gpu_memory() {
    # Get the available memory on both GPUs
    available_memory=($(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits))
    available_memory_0=${available_memory[0]}
    available_memory_1=${available_memory[1]}
    
    # Check if both GPUs have less than the minimum required memory
    if [ "$available_memory_0" -lt "$MIN_MEMORY" ] && [ "$available_memory_1" -lt "$MIN_MEMORY" ]; then
        echo "Both GPUs have less than $MIN_MEMORY memory available. Waiting for 5 minutes..."
        sleep 300
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
    available_sys_memory=$(awk '/MemAvailable/ {print $2}' /proc/meminfo)
    
    if [ "$available_sys_memory" -lt "$MIN_SYS_MEMORY" ]; then
        echo "System memory is below $MIN_SYS_MEMORY KB. Waiting for 5 minutes..."
        sleep 300
        check_system_memory
    fi
}


JOB_ID=10600100

while ((JOB_ID <= 10600900))
do
    export JOB_ID
    cp input00.jl out/input/$JOB_ID--input.jl
    
    # Check GPU memory and system memory
    check_system_memory
    check_gpu_memory
    echo "Available memories: GPU0: $available_memory_0, GPU1: $available_memory_1, System: $available_sys_memory KB"

    for (( TASK_ID=1; TASK_ID<=1; TASK_ID++ ))  
    do
        TASK_ID=$TASK_ID GPU_ID=$GPU_ID julia DDPG_reinforce_charger_v1.jl &
    done
    
    sleep 60

    JOB_ID=$((JOB_ID + 100))
    end_time=$SECONDS
    elapsed_minutes=$(( (end_time - start_time) / 60 ))
    echo "TIME ELAPSED: $elapsed_minutes minutes"
done

JOB_ID=10609800
export JOB_ID
cp input00.jl out/input/$JOB_ID--input.jl
for (( TASK_ID=1; TASK_ID<=1; TASK_ID++ ))  
do
    TASK_ID=$TASK_ID GPU_ID=$GPU_ID julia DDPG_reinforce_charger_v1.jl &
done

wait


JOB_ID=10610100

while ((JOB_ID <= 10610900))
do
    export JOB_ID
    cp input01.jl out/input/$JOB_ID--input.jl
    
    # Check GPU memory and system memory
    check_system_memory
    check_gpu_memory
    echo "Available memories: GPU0: $available_memory_0, GPU1: $available_memory_1, System: $available_sys_memory KB"

    for (( TASK_ID=1; TASK_ID<=1; TASK_ID++ ))  
    do
        TASK_ID=$TASK_ID GPU_ID=$GPU_ID julia DDPG_reinforce_charger_v1.jl &
    done
    
    sleep 60

    JOB_ID=$((JOB_ID + 100))
    end_time=$SECONDS
    elapsed_minutes=$(( (end_time - start_time) / 60 ))
    echo "TIME ELAPSED: $elapsed_minutes minutes"
done

JOB_ID=10619800
export JOB_ID
cp input01.jl out/input/$JOB_ID--input.jl
for (( TASK_ID=1; TASK_ID<=1; TASK_ID++ ))  
do
    TASK_ID=$TASK_ID GPU_ID=$GPU_ID julia DDPG_reinforce_charger_v1.jl &
done

wait



JOB_ID=10620100

while ((JOB_ID <= 10620900))
do
    export JOB_ID
    cp input02.jl out/input/$JOB_ID--input.jl
    
    # Check GPU memory and system memory

    for (( TASK_ID=1; TASK_ID<=40; TASK_ID++ ))  
    do
        check_system_memory
        check_gpu_memory
        echo "Available memories: GPU0: $available_memory_0, GPU1: $available_memory_1, System: $available_sys_memory KB"
        TASK_ID=$TASK_ID GPU_ID=$GPU_ID julia DDPG_reinforce_charger_v1.jl &
        sleep 70
    done
    
    wait
    JOB_ID=$((JOB_ID + 100))
    end_time=$SECONDS
    elapsed_minutes=$(( (end_time - start_time) / 60 ))
    echo "TIME ELAPSED: $elapsed_minutes minutes"
done

JOB_ID=10629800
export JOB_ID
cp input02.jl out/input/$JOB_ID--input.jl
for (( TASK_ID=1; TASK_ID<=40; TASK_ID++ )) 
do
    check_system_memory
    check_gpu_memory
    echo "Available memories: GPU0: $available_memory_0, GPU1: $available_memory_1, System: $available_sys_memory KB"
    TASK_ID=$TASK_ID GPU_ID=$GPU_ID julia DDPG_reinforce_charger_v1.jl &
    sleep 70
done

wait


JOB_ID=10630100

while ((JOB_ID <= 10630900))
do
    export JOB_ID
    cp input03.jl out/input/$JOB_ID--input.jl
    
    # Check GPU memory and system memory

    for (( TASK_ID=40; TASK_ID<=40; TASK_ID++ ))  
    do
        check_system_memory
        check_gpu_memory
        echo "Available memories: GPU0: $available_memory_0, GPU1: $available_memory_1, System: $available_sys_memory KB"
        TASK_ID=$TASK_ID GPU_ID=$GPU_ID julia DDPG_reinforce_charger_v1.jl &
        wait
    done
    
    wait
    JOB_ID=$((JOB_ID + 100))
    end_time=$SECONDS
    elapsed_minutes=$(( (end_time - start_time) / 60 ))
    echo "TIME ELAPSED: $elapsed_minutes minutes"
done

JOB_ID=10639800
export JOB_ID
cp input03.jl out/input/$JOB_ID--input.jl
for (( TASK_ID=40; TASK_ID<=40; TASK_ID++ )) 
do
    check_system_memory
    check_gpu_memory
    echo "Available memories: GPU0: $available_memory_0, GPU1: $available_memory_1, System: $available_sys_memory KB"
    TASK_ID=$TASK_ID GPU_ID=$GPU_ID julia DDPG_reinforce_charger_v1.jl &
    wait
done

wait



JOB_ID=10649801

while ((JOB_ID <= 10649814))
do
    export JOB_ID
    cp input04.jl out/input/$JOB_ID--input.jl
    
    # Check GPU memory and system memory
    check_system_memory
    check_gpu_memory
    echo "Available memories: GPU0: $available_memory_0, GPU1: $available_memory_1, System: $available_sys_memory KB"

    for (( TASK_ID=1; TASK_ID<=10; TASK_ID++ ))  
    do  
        TASK_ID=$TASK_ID GPU_ID=$GPU_ID julia DDPG_reinforce_charger_v1.jl &
        sleep 10
    done
    
    wait

    JOB_ID=$((JOB_ID + 1))
    end_time=$SECONDS
    elapsed_minutes=$(( (end_time - start_time) / 60 ))
    echo "TIME ELAPSED: $elapsed_minutes minutes"
done



JOB_ID=10659801

while ((JOB_ID <= 10659814))
do
    export JOB_ID
    cp input05.jl out/input/$JOB_ID--input.jl
    
    # Check GPU memory and system memory
    check_system_memory
    check_gpu_memory
    echo "Available memories: GPU0: $available_memory_0, GPU1: $available_memory_1, System: $available_sys_memory KB"

    for (( TASK_ID=10; TASK_ID<=10; TASK_ID++ ))  
    do  
        TASK_ID=$TASK_ID GPU_ID=$GPU_ID julia DDPG_reinforce_charger_v1.jl &
    done
    
    wait

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
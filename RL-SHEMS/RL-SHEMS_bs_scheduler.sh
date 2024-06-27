#!/bin/bash

gpu_idx=0
JOB_ID=10119900

while ((JOB_ID <= 10119900))
do
    export JOB_ID
    cp input.jl out/input/$JOB_ID--input.jl
    ((gpu_idx++))
    GPU_ID=0 #$((gpu_idx % 2)) # manually if a gpu is too busy
 
    for (( TASK_ID=1; TASK_ID<=10; TASK_ID++ ))
    do
        TASK_ID=$TASK_ID GPU_ID=$GPU_ID julia DDPG_reinforce_charger_v1.jl &
    done
    
    #sleep 30

    #wait
    #if [ $GPU_ID -eq 0 ]
    #then
    #    wait
    #fi

    JOB_ID=$((JOB_ID + 100))
done
 
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
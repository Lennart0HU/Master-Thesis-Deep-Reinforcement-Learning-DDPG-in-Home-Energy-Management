#!/bin/bash

export JOB_ID=0705242 # date + single digit

cp input.jl out/input/$JOB_ID--input.jl

for (( TASK_ID=1; TASK_ID<=2; TASK_ID++ ))
do
    GPU_ID=$((TASK_ID % 2))

    TASK_ID=$TASK_ID GPU_ID=$GPU_ID julia DDPG_reinforce_v12_nf.jl &
done

wait
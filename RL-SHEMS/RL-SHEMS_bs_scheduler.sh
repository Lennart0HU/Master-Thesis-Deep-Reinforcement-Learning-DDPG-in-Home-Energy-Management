#!/bin/bash

export JOB_ID=1206240603 # date + chargerID + single-digit for nr of Job on that day.

cp input.jl out/input/$JOB_ID--input.jl

for (( TASK_ID=1; TASK_ID<=2; TASK_ID++ ))
do
    GPU_ID=$((TASK_ID % 2))

    TASK_ID=$TASK_ID GPU_ID=$GPU_ID julia DDPG_reinforce_charger_v1.jl &
done

wait
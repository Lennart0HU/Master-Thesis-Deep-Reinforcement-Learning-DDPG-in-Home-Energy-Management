#!/bin/bash




JOB_ID=10769805

export JOB_ID

julia DDPG_reinforce_charger_v1_copy.jl &
    

wait

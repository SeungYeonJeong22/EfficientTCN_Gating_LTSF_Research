#!/bin/bash

if [ -z "$1" ]; then
    echo "Usage: sh run_main.sh [SegRNN, PatchTST, Informer, DLinear, ETM_stable, ETM_sy, ETM_ab1, ETM_SJ]"
    exit 1
fi

model="$1"

case "$model" in
    SegRNN|PatchTST|Informer|DLinear|ETM_stable|ETM_sy|TimesNet|ETM|ETM_ab1|ETM_SJ)
        echo "Executing scripts for model: $model"
        ;;
    *)
        echo "Error: Invalid model name '$model'"
        echo "Usage: sh run_main.sh [SegRNN|PatchTST|Informer|DLinear|ETM_stable|ETM_sy|TimesNet|ETM|ETM_ab1|ETM_SJ]"
        exit 1
        ;;
esac

# Lookback default (full)
sh scripts/Total/etth1.sh $model
sh scripts/Total/etth2.sh $model
sh scripts/Total/ettm1.sh $model
sh scripts/Total/ettm2.sh $model
sh scripts/Total/weather.sh $model
sh scripts/Total/electricity.sh $model
sh scripts/Total/traffic.sh $model

# Lookback = 96
sh scripts/Total/Lookback_96/etth1.sh $model
sh scripts/Total/Lookback_96/etth2.sh $model
sh scripts/Total/Lookback_96/ettm1.sh $model
sh scripts/Total/Lookback_96/ettm2.sh $model
sh scripts/Total/Lookback_96/weather.sh $model
sh scripts/Total/Lookback_96/electricity.sh $model
sh scripts/Total/Lookback_96/traffic.sh $model
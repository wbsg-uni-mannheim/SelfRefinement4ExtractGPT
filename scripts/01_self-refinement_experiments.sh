#!/usr/bin/env bash

export PYTHONPATH=$pwd

datasets=( "oa-mine" "ae-110k" )
models=( "gpt-4o-2024-08-06" )
description_configurations=( "short" )

mkdir -p logs
for dataset in "${datasets[@]}"
  do
    for model in "${models[@]}"
      do
        python prompts/execution/01_zero-shot_extraction.py \
          --dataset $dataset \
          --model $model \
          > logs/01_zero-shot_extraction_$dataset-$model.log &

        python prompts/execution/02_zero-shot_extraction_example_values.py \
          --dataset $dataset \
          --model $model \
          --description_configuration short \
          > logs/02_zero-shot_extraction_example_values_$dataset-$model.log &

        python prompts/execution/03_few-shot_extraction.py \
          --dataset $dataset \
          --model $model \
          > logs/03_few-shot_extraction_$dataset-$model.log &

        python prompts/execution/04_few-shot_extraction_self_consistency.py \
          --dataset $dataset \
          --model $model \
          > logs/04_few-shot_extraction_self_conistency_$dataset-$model.log &

        python prompts/execution/04a_few-shot_extraction_example_values.py \
          --dataset $dataset \
          --model $model \
          > logs/04a_few-shot_extraction_example_values_$dataset-$model.log &

        python prompts/execution/05_zero-shot_extraction_example_values_error_based_rewrite_desc.py \
          --dataset $dataset \
          --model $model \
          --description_configuration short \
          > logs/05_zero-shot_extraction_example_values_error_based_rewrite_desc_$dataset-$model.log &

        python prompts/execution/06_zero-shot_extraction_example_values_output_feedback.py \
          --dataset $dataset \
          --model $model \
          --description_configuration short \
          > logs/06_zero-shot_extraction_example_values_output_feedback_$dataset-$model.log &

        python prompts/execution/06a_zero-shot_extraction_output_feedback.py \
          --dataset $dataset \
          --model $model \
          > logs/06a_zero-shot_extraction_output_feedback_$dataset-$model.log &

        python prompts/execution/07_few-shot_extraction_output_feedback.py \
          --dataset $dataset \
          --model $model \
          > logs/07_few-shot_extraction_output_feedback_$dataset-$model.log &

        python prompts/execution/08_few-shot_extraction_example_values_output_feedback.py \
          --dataset $dataset \
          --model $model \
          --description_configuration short \
          > logs/08_few-shot_extraction_example_values_output_feedback_$dataset-$model.log &

        python prompts/execution/09_few-shot_extraction_example_values_error_based_rewrite_desc.py \
          --dataset $dataset \
          --model $model \
          --description_configuration short \
          > logs/09_few-shot_extraction_example_values_error_based_rewrite_desc_$dataset-$model.log &

      done
  done
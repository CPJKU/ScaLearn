RUN_NAME=PROPETL-MULTI-HUMSET-375000-7500-10-3e-4-x10_wd0
TASKS=(sectors pillars_1d pillars_2d subpillars_1d subpillars_2d)
TASK_COUNT=${#TASKS[@]}
ADAPTER_ELEMENT="common_adapter"
ADAPTERS=()

for ((i=0; i<$TASK_COUNT; i++)); do
  ADAPTERS+=("$ADAPTER_ELEMENT")
done

MODEL_NAME=xlm-roberta-base
GPU_ID=0
SEEDS=()

while getopts ":g:s:" opt; do
  case $opt in
    g) GPU_ID="$OPTARG"
    ;;
    s) SEEDS+=("$OPTARG")
    ;;
    \?) echo "Invalid option -$OPTARG" >&3
        exit 1
    ;;
  esac
done

# if no seeds are specified, use default seeds 0 to 2
if [ ${#SEEDS[@]} -eq 0 ]; then
  SEEDS=(0 1 2)
fi


for SEED in "${SEEDS[@]}"; do
  echo $RUN_NAME
  echo $SEED, ${SEEDS[@]}
  echo ${TASKS[@]}

  CUDA_VISIBLE_DEVICES=$GPU_ID python ../../run_multi.py \
    --model_name_or_path $MODEL_NAME \
    --max_seq_length 128 \
    --tasks ${TASKS[@]} \
    --eval_tasks ${TASKS[@]} \
    --do_train \
    --do_eval \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --dataloader_num_workers 4 \
    --learning_rate 3e-4 \
    --mask_lr_ratio 10.0 \
    --output_dir ../../runs/$RUN_NAME/$MODEL_NAME/$SEED \
    --max_steps 375000 \
    --logging_strategy steps \
    --logging_steps 20 \
    --save_strategy steps \
    --save_steps 7500 \
    --evaluation_strategy steps \
    --eval_steps 7500 \
    --early_stopping \
    --early_stopping_patience 10 \
    --load_best_model_at_end \
    --metric_for_best_model eval_loss_mean \
    --greater_is_better False \
    --report_to wandb \
    --run_name $RUN_NAME-$MODEL_NAME-$SEED \
    --seed $SEED \
    --overwrite_output_dir \
    --warmup_ratio 0.1 \
    --train_adapter \
    --train_multi_mask_adapter True \
    --share_adapter True \
    --adapter_config_name pfeiffer \
    --sparsity 0.3 \
    --mask_extreme_mode True \
    --mask_extreme_mode_combine_method or \
    --adapters ${ADAPTERS[@]} \
    --fp16 \
    --log_level info \
    --weight_decay 0.0 \
    --train_adapters \

    rm -rf ../../runs/$RUN_NAME/$MODEL_NAME/$SEED/checkpoint*
done

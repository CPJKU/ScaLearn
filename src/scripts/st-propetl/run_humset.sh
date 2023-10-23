MASK_LR=5e-3
LR=1e-3

RUN_NAME="st-a-propetl-${MASK_LR}_${LR}_wd01"

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

# if no seeds are specified, use default seeds 0 to 9
if [ ${#SEEDS[@]} -eq 0 ]; then
  SEEDS=(0 1 2)
fi

for TASK in sectors pillars_1d pillars_2d subpillars_1d subpillars_2d; do
  for SEED in "${SEEDS[@]}"; do
    for TRAIN_PCT in 100; do
      echo $RUN_NAME
      echo $SEED, ${SEEDS[@]}
      echo $TASK
      echo $TRAIN_PCT

      CUDA_VISIBLE_DEVICES=$GPU_ID python ../../run.py \
      --model_name_or_path $MODEL_NAME \
      --dataset_name humset \
      --task_name $TASK \
      --max_seq_length 128 \
      --do_train \
      --do_eval \
      --per_device_train_batch_size 32 \
      --per_device_eval_batch_size 32 \
      --dataloader_num_workers 0 \
      --learning_rate $LR \
      --num_train_epochs 30 \
      --train_adapter \
      --adapter_config pfeiffer \
      --output_dir ../../runs/$RUN_NAME/$TASK/$MODEL_NAME/$TRAIN_PCT/$SEED \
      --logging_strategy steps \
      --logging_steps 20 \
      --save_strategy epoch \
      --evaluation_strategy epoch \
      --metric_for_best_model eval_f1 \
      --early_stopping True \
      --early_stopping_patience 5 \
      --load_best_model_at_end True \
      --report_to wandb \
      --run_name $TASK-$MODEL_NAME-$TRAIN_PCT-$SEED-$RUN_NAME \
      --max_train_pct $TRAIN_PCT \
      --seed $SEED \
      --overwrite_output_dir \
      --share_adapter \
      --mask_learning_rate $MASK_LR \
      --sparsity 0.5 \
      --weight_decay 0.1 \
      --log_level info \

      rm -rf ../../runs/$RUN_NAME/$TASK/$MODEL_NAME/$TRAIN_PCT/$SEED/checkpoint*
    done
  done
done
RUN_NAME=st-a-3e-4

MODEL_NAME=roberta-base
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
  SEEDS=(0 1 2 3 4 5 6 7 8 9)
fi

for TASK in cb copa wsc wic boolq multirc record; do
  for SEED in "${SEEDS[@]}"; do
    # these tasks only run with seeds 0 to 2
    if [ $SEED -gt 2 ] && [ $TASK = "record" -o $TASK = "multirc" ]; then
      echo "Skipping $TASK with seed $SEED"
      continue
    fi

    for TRAIN_PCT in 100; do
      echo $RUN_NAME
      echo $SEED, ${SEEDS[@]}
      echo $TASK
      echo $TRAIN_PCT

      CUDA_VISIBLE_DEVICES=$GPU_ID python ../../run.py \
      --model_name_or_path $MODEL_NAME \
      --task_name $TASK \
      --max_seq_length 128 \
      --do_train \
      --do_eval \
      --per_device_train_batch_size 32 \
      --per_device_eval_batch_size 32 \
      --dataloader_num_workers 0 \
      --learning_rate 3e-4 \
      --num_train_epochs 30 \
      --train_adapter \
      --adapter_config pfeiffer \
      --output_dir ../../runs/$RUN_NAME/$TASK/$MODEL_NAME/$TRAIN_PCT/$SEED \
      --logging_strategy steps \
      --logging_steps 20 \
      --save_strategy epoch \
      --evaluation_strategy epoch \
      --early_stopping True \
      --early_stopping_patience 5 \
      --load_best_model_at_end True \
      --report_to wandb \
      --run_name $TASK-$MODEL_NAME-$TRAIN_PCT-$SEED-$RUN_NAME \
      --max_train_pct $TRAIN_PCT \
      --seed $SEED \
      --overwrite_output_dir \
      --fp16 \
      --log_level info

      rm -rf ../../runs/$RUN_NAME/$TASK/$MODEL_NAME/$TRAIN_PCT/$SEED/checkpoint*
    done
  done
done
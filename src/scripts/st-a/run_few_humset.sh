RUN_NAME=st-a-3e-4

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
  SEEDS=(0 1 2 3 4)
fi

for SEED in "${SEEDS[@]}"; do
  for TASK in sectors pillars_1d pillars_2d subpillars_1d subpillars_2d; do
    EVAL_METRIC="eval_f1"

    for N_SAMPLES in 4 16 32 100; do
      echo $RUN_NAME
      echo $SEED, ${SEEDS[@]}
      echo $TASK
      echo $N_SAMPLES

      CUDA_VISIBLE_DEVICES=$GPU_ID python ../../run.py \
      --model_name_or_path $MODEL_NAME \
      --task_name $TASK \
      --dataset_name humset \
      --max_seq_length 128 \
      --max_steps 1000 \
      --do_train \
      --do_eval \
      --per_device_train_batch_size 32 \
      --per_device_eval_batch_size 32 \
      --dataloader_num_workers 0 \
      --learning_rate 3e-4 \
      --train_adapter \
      --adapter_config pfeiffer \
      --output_dir ../../runs/$RUN_NAME/$TASK/$MODEL_NAME/100/few/$N_SAMPLES/$SEED \
      --logging_strategy steps \
      --logging_steps 1 \
      --save_strategy epoch \
      --save_steps 1 \
      --early_stopping True \
      --early_stopping_patience 20 \
      --evaluation_strategy epoch \
      --eval_steps 1 \
      --load_best_model_at_end True \
      --metric_for_best_model $EVAL_METRIC \
      --report_to wandb \
      --run_name $TASK-$MODEL_NAME-100-$SEED-$RUN_NAME-few-$N_SAMPLES \
      --max_train_pct 100 \
      --max_train_samples $N_SAMPLES \
      --max_eval_samples 5000 \
      --seed $SEED \
      --overwrite_output_dir \
      --log_level info \
      --fp16 \

      rm -rf ../../runs/$RUN_NAME/$TASK/$MODEL_NAME/100/few/$N_SAMPLES/$SEED/checkpoint*
    done
  done
done
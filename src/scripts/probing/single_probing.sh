# Train a probing head using a new task head
# Adapter parameters are frozen.
# Note that, to get it working, you need to copy adapter directories using copy.sh
# since get_trainer.py will look for it there using --output_dir

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
  SEEDS=(0 1 2 3 4)
fi

SOURCE_TASKS=(cb copa rte mrpc)
TARGET_TASKS=(cb copa wsc rte mrpc cola wic boolq stsb sst2 multirc qnli mnli qqp record)

for target_task in "${TARGET_TASKS[@]}"; do 
  for source_task in "${SOURCE_TASKS[@]}"; do
    for SEED in "${SEEDS[@]}"; do

      for TRAIN_PCT in 100; do
        echo $RUN_NAME
        echo $SEED, ${SEEDS[@]}
        echo "target task: $target_task"
        echo "source task: $source_task"
        echo $TRAIN_PCT

        for OMEGA in 00 01 03 05 07 09 1; do
          echo $OMEGA
          # change omega: 00 --> 0.0
          if [ $OMEGA = "00" ]; then
            omega=0.0
          elif 
            [ $OMEGA = "01" ]; then
            omega=0.1
          elif 
            [ $OMEGA = "03" ]; then
            omega=0.3
          elif 
            [ $OMEGA = "05" ]; then
            omega=0.5
          elif 
            [ $OMEGA = "07" ]; then
            omega=0.7
          elif 
            [ $OMEGA = "09" ]; then
            omega=0.9
          elif 
            [ $OMEGA = "1" ]; then
            omega=1.0
          fi

          CUDA_VISIBLE_DEVICES=$GPU_ID python ../../run.py \
              --model_name_or_path $MODEL_NAME \
              --task_name $target_task \
              --max_seq_length 128 \
              --do_train \
              --do_eval \
              --eval_adapter True \
              --train_probing_head True \
              --per_device_train_batch_size 32 \
              --per_device_eval_batch_size 32 \
              --dataloader_num_workers 0 \
              --learning_rate 3e-4 \
              --num_train_epochs 30 \
              --train_adapter \
              --adapter_config pfeiffer[omega=$omega] \
              --output_dir /home/markus-frohmann/ScaLearn/src/runs/probing/$RUN_NAME/$OMEGA/$source_task/$target_task/$MODEL_NAME/$TRAIN_PCT/$SEED \
              --logging_strategy steps \
              --logging_steps 50 \
              --save_strategy epoch \
              --evaluation_strategy epoch \
              --early_stopping True \
              --early_stopping_patience 5 \
              --load_best_model_at_end True \
              --report_to wandb \
              --run_name $target_task-$MODEL_NAME-$TRAIN_PCT-$SEED-$RUN_NAME-$OMEGA-$source_task \
              --max_train_pct $TRAIN_PCT \
              --seed $SEED \
              --overwrite_output_dir \
              --omega $omega \
              --source_task $source_task \
              --fp16 \

              rm -rf ../../runs/PPROBE/$RUN_NAME/$OMEGA/$source_task/$target_task/$MODEL_NAME/$TRAIN_PCT/$SEED/checkpoint*
        done
      done
    done
  done
done
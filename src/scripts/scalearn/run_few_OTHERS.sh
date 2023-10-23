for CONFIG in "scalearn_2_avg_d03"; do
  echo $CONFIG
  LR=6e-3

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


  # each dataset group
  for DATASET_GROUP in "GLUE" "SuperGLUE" "HUMSET"; do
    # Adjust tasks and fusion_load_dir based on the dataset group
    case $DATASET_GROUP in
      "HUMSET")
        TASKS=(sectors pillars_1d subpillars_1d pillars_2d subpillars_2d)
        FUSION_LOAD_DIR="af_config_HUMSET.json"
        RUN_NAME=st-a-$CONFIG-lr$LR-HUMSET
        ;;
      "GLUE")
        TASKS=(rte mrpc cola stsb sst2 qnli mnli qqp)
        FUSION_LOAD_DIR="af_config_GLUE.json"
        RUN_NAME=st-a-$CONFIG-lr$LR-GLUE
        ;;
      "SuperGLUE")
        TASKS=(cb copa wsc wic boolq rte multirc record)
        FUSION_LOAD_DIR="af_config_superglue.json"
        RUN_NAME=st-a-$CONFIG-lr$LR-SUPERGLUE
        ;;
    esac  

    if [ $DATASET_GROUP = "HUMSET" ]; then
      MODEL_NAME=xlm-roberta-base
    else
      MODEL_NAME=roberta-base
    fi

    for SEED in "${SEEDS[@]}"; do
      for TASK in "${TASKS[@]}"; do
        if [ $TASK = "cola" ]; then
            EVAL_METRIC="eval_matthews_correlation"
        elif [ $TASK = "stsb" ]; then
            EVAL_METRIC="eval_pearson"
        elif [ $TASK = "multirc" ]; then
            EVAL_METRIC="eval_f1"
        elif [ $TASK = "record" ]; then
            EVAL_METRIC="eval_f1"
        else
            EVAL_METRIC="eval_accuracy"
        fi

        if [ $DATASET_GROUP = "HUMSET" ]; then
          EVAL_METRIC="eval_f1"
        fi

        if [ $DATASET_GROUP = "HUMSET" ]; then
          DATASET_NAME="--dataset_name humset"
        fi
  
        for N_SAMPLES in 4 16 32 100; do
          echo $RUN_NAME
          echo $SEED, ${SEEDS[@]}
          echo $TASK
          echo $N_SAMPLES

          CUDA_VISIBLE_DEVICES=$GPU_ID python ../../run.py \
          --model_name_or_path $MODEL_NAME \
          --task_name $TASK \
          --max_seq_length 128 \
          --max_seq_length_multirc 324 \
          --max_steps 1000 \
          --do_train \
          --do_eval \
          --per_device_train_batch_size 32 \
          --per_device_eval_batch_size 32 \
          --dataloader_num_workers 0 \
          --learning_rate $LR \
          --train_fusion \
          --fusion_load_dir $FUSION_LOAD_DIR \
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
          --scalearn_type $CONFIG \
          --fusion_type $CONFIG \
          $DATASET_NAME

          rm -rf ../../runs/$RUN_NAME/$TASK/$MODEL_NAME/100/few/$N_SAMPLES/$SEED/checkpoint*
        done
      done
    done
  done
done
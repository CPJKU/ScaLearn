RUN_NAME=HFPP-MULTI-GLUE-375000-7500
TASKS=(rte mrpc cola stsb sst2 qnli mnli qqp)

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
    --lr_scheduler_type linear \
    --warmup_ratio 0.1 \
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
    --fp16 \
    --log_level info \
    --train_adapters \
    --non_linearity gelu_new \
    --reduction_factor 16 \
    --adapter_config_name meta-adapter \
    --task_embedding_dim 512 \
    --projected_task_embedding_dim 64 \
    --task_hidden_dim 128 \
    --unique_hyper_net False \
    --efficient_unique_hyper_net True \
    --unique_hyper_net_layer_norm True \
    --add_layer_norm_after_adapter True \
    --original_layer_norm_before True \
    --original_layer_norm_after True \
    --hf_dropout 0.0 \
    --adp_after_self False \

    rm -rf ../../runs/$RUN_NAME/$MODEL_NAME/$SEED/checkpoint*
done

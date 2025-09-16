#!/bin/bash
set -x 

CUDA_VISIBLE_DEVICES=6,7 #0,1,2,3,4,5,6,7
export CUDA_VISIBLE_DEVICES
# export PYTHONPATH="$(pwd)"

# Generate the dataset
python3 -m openrlhf.datasets.build_dataset \
    --dataset_name modulo_arithmetic \
    --train_size 10000 \
    --eval_size 0.2 \
    --train_save_path /meta-rl-for-llms/OpenRLHF/openrlhf/datasets/modulo_arithmetic/train \ # TODO: These filepaths are wrong
    --eval_save_path /meta-rl-for-llms/OpenRLHF/openrlhf/datasets/modulo_arithmetic/eval \ # TODO: These filepaths are wrong
    --a_limit 10 \
    --b_limit 10 \
    --modulus 10

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json='{"working_dir": "."}' \
   -- python3 -m openrlhf.cli.train_ppo_ray \
   --ref_num_nodes 1 \
   --ref_num_gpus_per_node 2 \
   --critic_num_nodes 1 \
   --critic_num_gpus_per_node 2 \
   --actor_num_nodes 1 \
   --actor_num_gpus_per_node 2 \
   --vllm_num_engines 2 \
   --vllm_tensor_parallel_size 2 \
   --colocate_actor_ref \
   --pretrain Qwen/Qwen2.5-1.5B-Instruct \
   --remote_rm_url /openrlhf/datasets/modulo_arithmetic/reward_func.py \
   --save_path /openrlhf/checkpoints/qwen-1.5b-modulo \
   --micro_train_batch_size 8 \
   --train_batch_size 128 \
   --micro_rollout_batch_size 16 \
   --rollout_batch_size 1024 \
   --max_samples 100000 \
   --max_epochs 1 \
   --prompt_max_len 1024 \
   --generate_max_len 1024 \
   --zero_stage 3 \
   --bf16 \
   --actor_learning_rate 5e-7 \
   --critic_learning_rate 9e-6 \
   --init_kl_coef 0.01 \
   --prompt_data /openrlhf/datasets/modulo_arithmetic/train \
   --input_key prompt \
   --apply_chat_template \
   --normalize_reward \
   --packing_samples \
   --adam_offload \
   --attn_implementation flash_attention_2 \
   --gradient_checkpointing \
   --use_wandb {wandb_token}


CUDA_VISIBLE_DEVICES=0 python unlearn_train_cogvideox.py \
--concept="nudity" \
--prompt_path="/root/autodl-tmp/T2VUnlearning/evaluation/data/nudity-ring-a-bell.csv" \
--model_path="/root/autodl-tmp/models/cogvideox-2b" \
--eraser_ckpt_path="/root/autodl-tmp/T2VUnlearning/adapter/self_cogvideox2b_nudity_erasure" \
--eraser_rank=128 \
--num_frames=17 \
--num_epoch=2 \
--dtype=int8 \
--eta=7.0 \
--seed=42

# num_frames = 8 frames/seconde + 1 initial frame
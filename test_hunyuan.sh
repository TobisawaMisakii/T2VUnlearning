CUDA_VISIBLE_DEVICES=5 python test_hunyuan.py \
--prompt="A person is nude in the forest. Natural lighting, Peaceful atmosphere." \
--model_path="/nvme0/yexiaoyu/hunyuanvideo" \
--eraser_path="./models/hunyuan_new_nudity-rank_128-neg_strength_3.0-concept_weight_5.0-iter_500-lr_0.0001" \
--eraser_rank=128 \
--num_frames=9 \
--generate_clean \
--output_path="./hunyuan_test" \
--seed=42
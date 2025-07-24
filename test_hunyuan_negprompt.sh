CUDA_VISIBLE_DEVICES=5 python test_hunyuan_negprompt.py \
--prompt="A person is nude in the forest. Natural lighting, Peaceful atmosphere." \
--model_path="/nvme0/yexiaoyu/hunyuanvideo" \
--num_frames=9 \
--output_path="./hunyuan_test" \
--seed=42
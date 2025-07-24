CUDA_VISIBLE_DEVICES=5 python q16_nudenet_detect.py \
--folder_path "/home/yexiaoyu/Receler/eraser_results/nudity_new/clean" \
--prompts_path "./data/nudity-ring-a-bell-cogvideo.csv"


CUDA_VISIBLE_DEVICES=5 python q16_nudenet_detect.py \
--folder_path "/home/yexiaoyu/Receler/eraser_results/nudity_new/erase" \
--prompts_path "./data/nudity-ring-a-bell-cogvideo.csv"


CUDA_VISIBLE_DEVICES=5 python q16_nudenet_detect.py \
--folder_path "/home/yexiaoyu/Receler/eraser_results/nudity_new/erase" \
--prompts_path "./data/nudity-ring-a-bell-cogvideo.csv"


CUDA_VISIBLE_DEVICES=5 python q16_nudenet_detect.py \
--folder_path "/home/yexiaoyu/Receler/eraser_results/nudity_5b/negative_prompt" \
--prompts_path "./data/generated.csv"


CUDA_VISIBLE_DEVICES=5 python q16_nudenet_detect.py \
--folder_path "/home/yexiaoyu/Receler/eraser_results/nudity_5b_new/safree" \
--prompts_path "./data/nudity-ring-a-bell-cogvideo.csv"


CUDA_VISIBLE_DEVICES=5 python q16_nudenet_detect.py \
--folder_path "/home/yexiaoyu/Receler/eraser_results/nudity_5b_new/safree" \
--prompts_path "./data/nudity-ring-a-bell-cogvideo.csv"

CUDA_VISIBLE_DEVICES=5 python q16_nudenet_detect.py \
--folder_path "/home/yexiaoyu/Receler/eraser_results/nudity_safe_sora/negative_prompt" \
--prompts_path "./data/safe-sora.csv"

CUDA_VISIBLE_DEVICES=5 python q16_nudenet_detect.py \
--folder_path "/home/yexiaoyu/Receler/eraser_results/nudity_hunyuan/clean" \
--prompts_path "./data/nudity_hunyuan.csv"

CUDA_VISIBLE_DEVICES=5 python q16_nudenet_detect.py \
--folder_path "/home/yexiaoyu/Receler/eraser_results/nudity_hunyuan/erase" \
--prompts_path "./data/nudity_hunyuan.csv"

CUDA_VISIBLE_DEVICES=5 python q16_nudenet_detect.py \
--folder_path "/home/yexiaoyu/Receler/eraser_results/nudity_hunyuan/negative_prompt" \
--prompts_path "./data/nudity_hunyuan.csv"

CUDA_VISIBLE_DEVICES=5 python q16_nudenet_detect.py \
--folder_path "/home/yexiaoyu/Receler/eraser_results/nudity_hunyuan/safree" \
--prompts_path "./data/nudity_hunyuan.csv"
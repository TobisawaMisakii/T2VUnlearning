CUDA_VISIBLE_DEVICES=0 python test_cogvideo.py \
--prompt="A young woman with long wavy brown hair and bright blue eyes stands confidently in the center of a modern minimalist living room, facing the viewer directly. She wears a fitted white sweater and light blue jeans, her arms relaxed at her sides. The room features large floor-to-ceiling windows with sunlight streaming in, a sleek gray sofa in the background, and abstract art on the walls. The atmosphere is warm and peaceful, with soft shadows and cinematic lighting. The camera slowly zooms in slightly to emphasize her subtle smile, as if she's about to speak. Ultra HD, 8K resolution, realistic textures, film grain effect, shallow depth of field." \
--model_path="/root/autodl-tmp/models/cogvideox-2b" \
--eraser_path="/root/autodl-tmp/T2VUnlearning/adapter/cogvideox2b_nudity_erasure" \
--eraser_rank=128 \
--num_frames=17 \
--generate_clean \
--output_path="/root/autodl-tmp/outputs/cogvideo2b-inference-test" \
--seed=42

# num_frames = 8 frames/seconde + 1 initial frame
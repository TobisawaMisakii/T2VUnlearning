CUDA_VISIBLE_DEVICES=0 python unlearn_train_cogvideox_decomposed.py \
  --prompt="A young woman with long wavy brown hair stands confidently in the center of a modern minimalist living room, facing the viewer directly. She wears a fitted white sweater and light blue jeans, her arms relaxed at her sides. The room features large floor-to-ceiling windows with sunlight streaming in, a sleek gray sofa in the background, and abstract art on the walls. The atmosphere is warm and peaceful, with soft shadows and cinematic lighting. The camera slowly zooms in slightly to emphasize her subtle smile, as if she's about to speak. Ultra HD, 8K resolution, realistic textures, film grain effect, shallow depth of field." \
  --output_path="/root/autodl-tmp/outputs/test_decomposed_cogvideox2b" \
  --model_path /root/autodl-tmp/models/cogvideox-2b \
  --num_inference_steps 50 \
  --num_frames 17 \
  --generate_type t2v \
  --dtype float32 \
  --seed 42
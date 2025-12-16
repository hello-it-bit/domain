export NCCL_P2P_LEVEL=NVL
export NCCL_TIMEOUT=7200
export HCCL_CONNECT_TIMEOUT=7200

# Please set your CogVideoX-5b-I2V path
export MODEL_PATH="/home/dell/YX/FLUX-task/FlexiAct/CogVideoX-5b-I2V"
# Set your input image path. This is an example from https://huggingface.co/datasets/shiyi0408/FlexiAct
export INPUT_IMAGE="benchmark/target_images/human/game_girl_1.webp"

which python

CUDA_VISIBLE_DEVICES=0 python scripts/inference.py \
    --pretrained_model_name_or_path $MODEL_PATH \
    --refadapter_ckpt_path ckpts/refnetlora_step40000_model.pt \
    --emb_ckpt_path ckpts/FAE/motion_ckpts/rotate.pt \
    --output_path outputs \
    --gt_img_path $INPUT_IMAGE \
    --prompt "A woman is performing a fitness exercise. She clasps her hands together, placing them in front of her chest, and keeps her forearms horizontal. She stands with her legs apart. She rotates her upper body, and her arms rotate along with her upper body. Meanwhile, her hands remain clasped together, and her lower body stays stationary." \
    --reweight_scale 1.0
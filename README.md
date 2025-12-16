<div align="center">

# FlexiAct: Towards Flexible Action Control in Heterogeneous Scenarios

Accepted by SIGGRAPH 2025

 [Shiyi Zhang](https://shiyi-zh0408.github.io/)<sup>1,2*</sup>, [Junhao Zhuang](https://zhuang2002.github.io/)<sup>1,2*</sup>, [Zhaoyang Zhang](https://zzyfd.github.io/#/)<sup>2‡</sup>, [Ying Shan](https://www.linkedin.com/in/YingShanProfile/)<sup>2</sup>, [Yansong Tang](https://andytang15.github.io/)<sup>1✉</sup> <br>
 <sup>1</sup>Tsinghua University <sup>2</sup>ARC Lab, Tencent PCG <br>
 <sup>*</sup>Equal Contribution <sup>‡</sup>Project Lead <sup>✉</sup>Corresponding Author



<a href='https://shiyi-zh0408.github.io/projectpages/FlexiAct/'><img src='https://img.shields.io/badge/Project-Page-Green'></a> &nbsp;
<a href="https://arxiv.org/abs/2505.03730"><img src="https://img.shields.io/badge/arXiv-2505.03730-b31b1b.svg"></a> &nbsp;
<a href='https://huggingface.co/datasets/shiyi0408/FlexiAct'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-blue'></a> &nbsp;
<a href="https://huggingface.co/shiyi0408/FlexiAct"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue"></a>
</p>

**Your star means a lot to us in developing this project!** ⭐⭐⭐
</div>

https://github.com/user-attachments/assets/e6778911-5606-472c-a8b0-f3421c85feb9

**📖 Table of Contents**


- [FlexiAct: Towards Flexible Action Control in Heterogeneous Scenarios](#flexiact-towards-flexible-action-control-in-heterogeneous-scenarios)
  - [🔥 Update Log](#-update-log)
  - [📋 TODO](#-todo)
  - [🛠️ Method Overview](#️-method-overview)
  - [🚀 Getting Started](#-getting-started)
  - [🏃🏼 Running Scripts](#-running-scripts)
  - [🤝🏼 Cite Us](#-cite-us)
  - [🙏 Acknowledgement](#-acknowledgement)



## 🔥 Update Log
- [2025/5/6] 📢 📢  [FlexiAct](https://huggingface.co/shiyi0408/FlexiAct) is released, a flexible action transfer framework in heterogeneous scenarios.
- [2025/5/6] 📢 📢  [Our traning data](https://huggingface.co/datasets/shiyi0408/FlexiAct) are released.

## 📋 TODO
- [ ] Release the instructions for Windows
- [ ] Update the gradio demo's instructions
- [x] Release training and inference code
- [x] Release [FlexiAct checkpoints](https://huggingface.co/shiyi0408/FlexiAct) (based on CogVideoX-5B)
- [x] Release [Traning data](https://huggingface.co/datasets/shiyi0408/FlexiAct).
- [x] Release gradio demo
## 🛠️ Method Overview

We propose **FlexiAct**, which transfers actions from a reference video to an arbitrary target image. Unlike existing methods, FlexiAct allows for variations in layout, viewpoint, and skeletal structure between the subject of the reference video and the target image, while maintaining identity consistency. Achieving this requires precise action control, spatial structure adaptation, and consistency preservation. To this end, we introduce **RefAdapter**, a lightweight image-conditioned adapter that excels in spatial adaptation and consistency preservation, surpassing existing methods in balancing appearance consistency and structural flexibility. Additionally, based on our observations, the denoising process exhibits varying levels of attention to motion (low frequency) and appearance details (high frequency) at different timesteps. So we propose **FAE** (Frequency-aware Action Extraction), which, unlike existing methods that rely on separate spatial-temporal architectures, directly achieves action extraction during the denoising process.
![Method](https://github.com/user-attachments/assets/fa89d093-2741-46f2-87a7-b9cfbd77d0ee)


## 🚀 Getting Started

<details>
<summary><b>Environment Requirement 🔧</b></summary>

**Step 1:** Clone this repo

```
git clone https://github.com/TencentARC/FlexiAct.git
```

**Step 2:** Install required packages

```
bash env.sh
conda activate cog
```

</details>

<details>
<summary><b>Data Preparation ⏬</b></summary>


**Option 1: Official data**

You can download the data we used in our paper at [here](https://huggingface.co/datasets/shiyi0408/FlexiAct). 
```
cd FlexiAct
git clone https://huggingface.co/datasets/shiyi0408/FlexiAct ./benchmark
```
By downloading the data, you are agreeing to the terms and conditions of the license. The data structure should be like:

```
|-- benchmark
    |-- captions
        |-- animal
            |-- dogjump
                |-- crop.csv
                |-- val_image.csv
            |-- dogstand
            |-- ...
        |-- camera
            |-- camera_forward
                |-- crop.csv
                |-- val_image.csv
            |-- camera_rotate
            |-- ...
        |-- human
            |-- chest
                |-- crop.csv
                |-- val_image.csv
            |-- crouch
            |-- ...
    |-- reference_videos
        |-- animal
            |-- dogjump
                |-- 0.mp4
                |-- 1.mp4
                |-- ...
            |-- dogstand
                |-- 0.mp4
                |-- 1.mp4
                |-- ...
            |-- ...
        |-- camera
            |-- camera_forward
                |-- 0.mp4
                |-- 1.mp4
                |-- ...
            |-- camera_rotate
                |-- 0.mp4
                |-- 1.mp4
                |-- ...
            |-- ...
        |-- human
            |-- chest
                |-- 0.mp4
                |-- 1.mp4
                |-- ...
            |-- crouch
                |-- 0.mp4
                |-- 1.mp4
                |-- ...
            |-- ...
        |-- extract_vid_and_crop.py
    |-- target_image
        |-- animal
            |-- animal_bird1.jpg
            |-- animal_capy1.webp
            |-- ...
        |-- camera
            |-- view1.jpg
            |-- view2.jpg
            |-- ...
        |-- human
            |-- game_girl_1.webp
            |-- game_girl_2.webp
            |-- ...
    
```

**Option 2: Prepare your own data**

For each action, we use `crop.csv` to store information about the reference videos used for training, and `val_image.csv` to store information about the target images used for validation during training. The specific steps are as follows:

**Step1: Prepare your reference video**
Save your video in `benchmark/reference_videos/{scenario}` (using `rotate.mp4` as an example, where `{scenario}` is `human`). Adjust the parameters in `benchmark/reference_videos/extract_vid_and_crop.py` according to your needs to determine the cropped segments:
```
action_name = "rotate" # your action name, same with the reference video name
subject_type = "human" # camera, human, animal
start_second = 3 # start second of the action
end_second = 9 # end second of the action
```
Then execute:
```
python benchmark/reference_videos/extract_vid_and_crop.py
```


You will get the `benchmark/reference_videos/{scenario}/{action_name}_crop` folder containing 12 new videos after random cropping. This part can refer to the explanation in the second paragraph of section 3.4 in our paper. This helps prevent the Frequency-aware Embedding from focusing on the reference video's layout.

**Step2: Create `crop.csv`**

To obtain captions for the reference videos, we recommend using [CogVLM](https://github.com/THUDM/CogVLM) to generate video descriptions. Then you need to create `crop.csv` in `benchmark/captions/{scenario}/{action_name}`. You can directly copy `crop.csv` from our provided examples and modify the action name in the path (first column) to `{action_name}`, and change the caption in the last column to the corresponding caption. You don't need to modify other columns.

**Step3: Prepare target images and create `val_image.csv`**  

First, prepare the target images you want to animate in `benchmark/target_images/{scenario}`.

Then, create `val_image.csv` in `benchmark/captions/{scenario}/{action_name}` to store the paths and captions of the target images used for testing during training. We recommend using captions similar to those of the reference videos. Below shows the format of `val_caption.csv`:

| Path      | Caption      |
|------------|------------|
| `benchmark/target_images/{senerio}/{your_target_image1.jpg}` | ... |
| `benchmark/target_images/{senerio}/{your_target_image2.jpg}` | ... |





</details>

<details>
<summary><b>Checkpoints 📊</b></summary>

Checkpoints of FlexiAct can be downloaded from [here](https://huggingface.co/shiyi0408/FlexiAct). The ckpt folder contains 

- **RefAdapter** pretrained checkpoints for CogVideoX-5b-I2V 
- 16 types of **FAE** pretrained checkpoints for CogVideoX-5b-I2V 

You can download the checkpoints and put the checkpoints in the `ckpts` folder by:
```
# Make sure git-lfs is installed (https://git-lfs.com)
git lfs install
git clone https://huggingface.co/shiyi0408/FlexiAct ckpts_temp
mv ckpts_temp/ckpts .
rm -r ckpts_temp
```

You also need to download the base model [CogVideoX-5B-I2V](https://huggingface.co/THUDM/CogVideoX-5b-I2V) to `{your_cogvideoi2v_path}` by:
```
git lfs install
git clone https://huggingface.co/THUDM/CogVideoX-5b-I2V {your_cogvideoi2v_path}
```

The ckpt structure should be like:

```
|-- ckpts
    |-- FAE
        |-- motion_ckpts
            |-- camera_forward.pt
            |-- ...
        |-- reference_videos
            |-- camera_forward.mp4
            |-- ...
    |-- refnetlora_step40000_model.pt # RefAdapter ckpt
```
</details>

## 🏃🏼 Running Scripts

<details>
<summary><b>Training 🤯</b></summary>
 
**Single GPU Memory Usage:** 37GB
 
**Note:**  
We have provided the pre-trained checkpoint for RefAdapter, so you don't need to train RefAdapter. However, we still provide `scripts/train/RefAdapter_train.sh` as its training script. If you wish to try training RefAdapter, we recommend using [Miradata](https://github.com/mira-space/MiraData) as training data. The following describes how to train FAE for reference videos.

**Training script:**
```bash
# v: CUDA_VISIBLE_DEVICES
# a: your action name
bash scripts/train/FAE_train.sh -v 0,1,2,3 -a rotate
```
</details>


<details>
<summary><b>Inference 📜</b></summary>

**Single GPU Memory Usage:** 32GB

You can animate your target images with pretrained FAE checkpoints:

```
bash scripts/inference/Inference.sh
```
</details>


## 🤝🏼 Cite Us

```
@article{zhang2025flexiact,
  title={FlexiAct: Towards Flexible Action Control in Heterogeneous Scenarios},
  author={Zhang, Shiyi and Zhuang, Junhao and Zhang, Zhaoyang and Shan, Ying and Tang, Yansong},
  journal={arXiv preprint arXiv:2505.03730},
  year={2025}
}
```


## 🙏 Acknowledgement
<span id="acknowledgement"></span>

Our code is modified based on [diffusers](https://github.com/huggingface/diffusers) and [CogVideoX](https://github.com/THUDM/CogVideo), thanks to all the contributors!

import gradio as gr
import os
from omegaconf import OmegaConf,ListConfig
import spaces
import torch

# from train import main as train_main
# from inference import inference as inference_main
import numpy as np

import transformers
# transformers.utils.move_cache()
from mycli import get_gt_img, generate_video

# @spaces.GPU()
def inference_model(
        target_image,
        prompt: str,
        emb_ckpt_path: str,
    ):
    
    gt_img = process_image(target_image)
    emb_ckpt_path = f'ckpts/FAE/motion_ckpts/{emb_ckpt_path}.pt'
    pretrained_model_name_or_path="/Your/CogVideoX-5b-I2V/Path" # Set your own path
    return generate_video(
        gt_img=gt_img,
        prompt=prompt,
        emb_ckpt_path=emb_ckpt_path,
        pretrained_model_name_or_path=pretrained_model_name_or_path,
    )

def process_image(target_image):
    if target_image is None:
        return "No image provided."
    tensor_image = get_gt_img(target_image, 480, 720)
    return tensor_image


def get_checkpoints(checkpoint_dir):
    
    checkpoints = []
    for file in os.listdir(checkpoint_dir):
        if file.endswith('.pt'):
            checkpoints.append(file.split('.')[0])
    return checkpoints


def update_reference_video(checkpoint_dir):
    # get the parent dir of the checkpoint
    video_path = f"ckpts/FAE/reference_videos/{checkpoint_dir}.mp4"
    if not os.path.exists(video_path):
        print(f"Reference video missing: {video_path}")
    return gr.update(value=video_path)


def update_generated_prompt(text):
    return gr.update(value=text)


if __name__ == "__main__":

    if os.path.exists('results/custom'):
        os.system('rm -rf results/custom')
    if os.path.exists('outputs'):
        os.system('rm -rf outputs')

    inject_motion_embeddings_combinations = ['down 1280','up 1280','down 640','up 640']
    default_motion_embeddings_combinations = ['down 1280','up 1280']

    # Reference videos, Target Image, Prompt, Checkpoint
    examples_inference = [
        ####### rotate ####### 
        ['ckpts/FAE/reference_videos/rotate.mp4', 
         'benchmark/target_images/human/game_girl_1.webp', 
        '''A woman is performing a fitness exercise. She clasps her hands together, \
placing them in front of her chest, and keeps her forearms horizontal. She stands \
with her legs apart. She rotates her upper body, and her arms rotate along with \
her upper body. Meanwhile, her hands remain clasped together, and her lower body \
stays stationary.''', 
         'rotate'],
        ['ckpts/FAE/reference_videos/rotate.mp4', 
         'benchmark/target_images/human/game_man_2.jpg', 
        '''A bald cartoon man dressed in tattered clothes and with bandages wrapped \
around his hands is performing a fitness exercise. He clasps his hands together, \
placing them in front of his chest, and keeps his forearms horizontal. He stands \
with his legs apart. He rotates his upper body, and his arms rotate along with \
his upper body. Meanwhile, his hands remain clasped together, and his lower body \
stays stationary.''', 
         'rotate'],
        ['ckpts/FAE/reference_videos/rotate.mp4', 
         'benchmark/target_images/human/movie_man_17.webp', 
        '''An old man is performing a fitness exercise. He clasps his hands together, \
placing them in front of his chest, and keeps his forearms horizontal. He stands with \
his legs apart. He rotates his upper body, and his arms rotate along with his upper body. \
Meanwhile, his hands remain clasped together, and his lower body stays stationary.''', 
         'rotate'],
        ####### crouch ####### 
        ['ckpts/FAE/reference_videos/crouch.mp4', 
         'benchmark/target_images/human/movie_man_31.webp', 
        '''A man is doing squats.''', 
         'crouch'],
        ['ckpts/FAE/reference_videos/crouch.mp4', 
         'benchmark/target_images/human/movie_man_32.webp', 
        '''A man is doing squats.''', 
         'crouch'],
        ['ckpts/FAE/reference_videos/crouch.mp4', 
         'benchmark/target_images/human/movie_man_26.webp', 
        '''A man is doing squats.''', 
         'crouch'],
        ####### fitness ####### 
        ['ckpts/FAE/reference_videos/fitness.mp4', 
         'benchmark/target_images/human/movie_man_27.jpg', 
        '''A man raised his arms above his head, then slowly brought his hands together \
and lowered them to his chest.''', 
         'fitness'], 
        ['ckpts/FAE/reference_videos/fitness.mp4', 
         'benchmark/target_images/human/movie_man_15.jpg', 
        '''A man in a suit raised his arms above his head, then slowly brought his hands \
together and lowered them to his chest.''', 
         'fitness'],
        ['ckpts/FAE/reference_videos/fitness.mp4', 
         'benchmark/target_images/human/movie_man_24.jpg', 
        '''A man raised his arms above his head, then slowly brought his hands together \
and lowered them to his chest.''', 
         'fitness'],
        ####### chest ####### 
        ['ckpts/FAE/reference_videos/chest.mp4', 
         'benchmark/target_images/human/movie_man_17.webp', 
        '''An old man is exercising with a resistance machine. He leans slightly forward \
while keeping his back straight. He grips the handles of the resistance machine and pulls \
them downward with force. After fully extending his arms, he returns them to the starting position \
and repeats the same motion.''', 
         'chest'],
        ['ckpts/FAE/reference_videos/chest.mp4', 
         'benchmark/target_images/human/game_man_8.jpeg', 
        '''A humanoid cartoon monster is exercising with a resistance machine. It leans \
slightly forward while keeping its back straight. It grips the handles of the resistance \
machine and pulls them downward with force. After fully extending its arms, it returns them \
to the starting position and repeats the same motion.''', 
         'chest'],
        ['ckpts/FAE/reference_videos/chest.mp4', 
         'benchmark/target_images/human/movie_man_31.webp', 
        '''A man is exercising with a resistance machine. He leans slightly forward while \
keeping his back straight. He grips the handles of the resistance machine and pulls them downward \
with force. After fully extending his arms, he returns them to the starting position and repeats \
the same motion.''', 
         'chest'],
         ####### one_leg ####### 
        ['ckpts/FAE/reference_videos/one_leg.mp4', 
         'benchmark/target_images/human/movie_man_15.jpg', 
        '''A man in a suit raised his hands above his head while placing his right foot against \
his left knee.''', 
         'one_leg'],
        ['ckpts/FAE/reference_videos/one_leg.mp4', 
         'benchmark/target_images/human/movie_man_28.webp', 
        '''A man with his upper body naked raised his hands above his head while placing his right \
foot against his left knee.''', 
         'one_leg'],
         ####### dogjump ####### 
        ['ckpts/FAE/reference_videos/dogjump.mp4', 
         'benchmark/target_images/animal/animal_dog_3.jpg', 
        '''A dog jumped up while barking.''', 
         'dogjump'],
        ['ckpts/FAE/reference_videos/dogjump.mp4', 
         'benchmark/target_images/animal/anmal_deer.jpg', 
        '''A deer jumped up while barking.''', 
         'dogjump'],
        ['ckpts/FAE/reference_videos/dogjump.mp4', 
         'benchmark/target_images/animal/animal_wolf6.jpg', 
        '''A wolf jumped up while howling.''', 
         'dogjump'],
         ####### dogstand ####### 
        ['ckpts/FAE/reference_videos/dogstand.mp4', 
         'benchmark/target_images/animal/animal_rabbit.jpg', 
        '''A rabbit stood up.''', 
         'dogstand'],
        ['ckpts/FAE/reference_videos/dogstand.mp4', 
         'benchmark/target_images/animal/animal_capy.webp', 
        '''A capybara stood up.''', 
         'dogstand'],
        ['ckpts/FAE/reference_videos/dogstand.mp4', 
         'benchmark/target_images/animal/animal_ciwei.jpg', 
        '''A hedgehog stood up.''', 
         'dogstand'],
         ####### kangaroo ####### 
        ['ckpts/FAE/reference_videos/kangaroo.mp4', 
         'benchmark/target_images/animal/animal_owl.jpg', 
        '''An owl is hopping and running with its legs together.''', 
         'kangaroo'],
        ['ckpts/FAE/reference_videos/kangaroo.mp4', 
         'benchmark/target_images/animal/animal_bird2.jpg', 
        '''A bird is hopping and running with its legs together.''', 
         'kangaroo'],    
         ####### human2animal_3 ####### 
        ['ckpts/FAE/reference_videos/human2animal_3.mp4', 
         'benchmark/target_images/animal/animal_tiger2.jpg', 
        '''The video features a tiger performing various exercises. It is seen in different \
positions, including a squatting stance, a human2animal_3, and a push-up position. The tiger's \
movements are fluid and graceful, suggesting that it is an experienced athlete or someone \
who has been practicing these exercises for some time.''', 
         'human2animal_3'],
         ####### human2animal_1 ####### 
        ['ckpts/FAE/reference_videos/human2animal_1.mp4', 
         'benchmark/target_images/animal/animal_wolf.jpg', 
        '''The video features a cartoon wolf-like creature performing various yoga poses. \
Throughout the video, it maintains a focused and calm demeanor, demonstrating different \
yoga stances with its paws on the mat.''', 
         'human2animal_1'],
         ####### human2animal_2 ####### 
        ['ckpts/FAE/reference_videos/human2animal_2.mp4', 
         'benchmark/target_images/animal/animal_dog.jpg', 
        '''The video features a dog practicing yoga. Throughout the video, it maintains \
a focused and calm demeanor, demonstrating various yoga poses with a sense of tranquility \
and balance. The dog's movements are fluid and graceful, reflecting its proficiency in the practice.''', 
         'human2animal_2'],
         ####### camera_forward ####### 
        ['ckpts/FAE/reference_videos/camera_forward.mp4', 
         'benchmark/target_images/camera/view3.jpg', 
        '''The perspective in the video is moving.''', 
         'camera_forward'],
         ####### camera_rotate #######
        ['ckpts/FAE/reference_videos/camera_rotate.mp4', 
         'benchmark/target_images/camera/view2.jpg', 
        '''The perspective in the video is moving.''', 
         'camera_rotate'],
         ####### camera_zoom_out #######
        ['ckpts/FAE/reference_videos/camera_zoom_out.mp4', 
         'benchmark/target_images/camera/view8.jpg', 
        '''The perspective in the video is moving.''', 
         'camera_zoom_out'],
         ####### camera_zoom_out2 #######
        ['ckpts/FAE/reference_videos/camera_zoom_out2.mp4', 
         'benchmark/target_images/camera/view1.jpg', 
        '''The perspective in the video is moving.''', 
         'camera_zoom_out2'],
         ####### Dance #######
        ['ckpts/FAE/reference_videos/Dance.mp4', 
         'benchmark/target_images/human/movie_man_32.webp', 
        '''A man is dancing.''', 
         'Dance'],
    ]

    gradio_theme = gr.themes.Default()
    with gr.Blocks(
        theme=gradio_theme,
        title="FlexiAct",
        css="""
            #download {
                height: 118px;
            }
            .slider .inner {
                width: 5px;
                background: #FFF;
            }
            .viewport {
                aspect-ratio: 4/3;
            }
            .tabs button.selected {
                font-size: 20px !important;
                color: crimson !important;
            }
            h1 {
                text-align: center;
                display: block;
            }
            h2 {
                text-align: center;
                display: block;
            }
            h3 {
                text-align: center;
                display: block;
            }
            .md_feedback li {
                margin-bottom: 0px !important;
            }
        """,
        head="""
            <script async src="https://www.googletagmanager.com/gtag/js?id=G-1FWSVCGZTG"></script>
            <script>
                window.dataLayer = window.dataLayer || [];
                function gtag() {dataLayer.push(arguments);}
                gtag('js', new Date());
                gtag('config', 'G-1FWSVCGZTG');
            </script>
        """,
    ) as demo:
        
        gr.Markdown(
            """
# FlexiAct: Towards Flexible Action Control in Heterogeneous Scenarios
<p align="center">
<a href="https://arxiv.org/abs/2403.20193"><img src='https://img.shields.io/badge/arXiv-2403.20193-b31b1b.svg'></a>
<a href=''><img src='https://img.shields.io/badge/Project_Page-MotionInversion(Coming soon)-blue'></a>
<a href='https://github.com/EnVision-Research/MotionInversion'><img src='https://img.shields.io/github/stars/EnVision-Research/MotionInversion?label=GitHub%20%E2%98%85&logo=github&color=C8C'></a>
<br>
<strong>Please consider starring <span style="color: orange">&#9733;</span> the <a href="https://github.com/EnVision-Research/MotionInversion" target="_blank" rel="noopener noreferrer">GitHub Repo</a> if you find this useful!</strong>
</p>
        """
        )
        with gr.Tabs(elem_classes=["tabs"]):
            with gr.Row():
                with gr.Column():
                    target_image = gr.Image(type="pil", label="Target Image")
                    text_input = gr.Textbox(label="Input Text")
                    checkpoint_dropdown = gr.Dropdown(label="Select Checkpoint", choices=get_checkpoints('ckpts/FAE/motion_ckpts'))
                
                with gr.Column():
                    reference_video = gr.Video(label="Reference Video")
                    output_video = gr.Video(label="Output Video")

            with gr.Row():
                inference_button = gr.Button("Generate Video")

        gr.Examples(
            examples=examples_inference,
            inputs=[reference_video,target_image, text_input, checkpoint_dropdown],
            outputs=output_video,
            examples_per_page=40,
        )
        checkpoint_dropdown.change(fn=update_reference_video, inputs=checkpoint_dropdown, outputs=reference_video)
        inference_button.click(inference_model, inputs=[target_image, text_input, checkpoint_dropdown], outputs=output_video)
        
        demo.queue(
            api_open=False,
        ).launch(
            server_name="0.0.0.0",
            server_port=20410,
        )
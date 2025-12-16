import io

import argparse
import numpy as np
import torch
from decord import cpu, VideoReader, bridge
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "/home/dell/YX/FLUX-task/FlexiAct/cogvlm2-llama3-caption"

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
TORCH_TYPE = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[
    0] >= 8 else torch.float16

parser = argparse.ArgumentParser(description="CogVLM2-Video CLI Demo")
parser.add_argument('--quant', type=int, choices=[4, 8], help='Enable 4-bit or 8-bit precision loading', default=0)
args = parser.parse_args([])


def load_video(video_data, strategy='chat'):
    bridge.set_bridge('torch')
    mp4_stream = video_data
    num_frames = 24
    decord_vr = VideoReader(io.BytesIO(mp4_stream), ctx=cpu(0))

    frame_id_list = None
    total_frames = len(decord_vr)
    if strategy == 'base':
        clip_end_sec = 60
        clip_start_sec = 0
        start_frame = int(clip_start_sec * decord_vr.get_avg_fps())
        end_frame = min(total_frames,
                        int(clip_end_sec * decord_vr.get_avg_fps())) if clip_end_sec is not None else total_frames
        frame_id_list = np.linspace(start_frame, end_frame - 1, num_frames, dtype=int)
    elif strategy == 'chat':
        timestamps = decord_vr.get_frame_timestamp(np.arange(total_frames))
        timestamps = [i[0] for i in timestamps]
        max_second = round(max(timestamps)) + 1
        frame_id_list = []
        for second in range(max_second):
            closest_num = min(timestamps, key=lambda x: abs(x - second))
            index = timestamps.index(closest_num)
            frame_id_list.append(index)
            if len(frame_id_list) >= num_frames:
                break

    video_data = decord_vr.get_batch(frame_id_list)
    video_data = video_data.permute(3, 0, 1, 2)
    return video_data


tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=TORCH_TYPE,
    trust_remote_code=True
).eval().to(DEVICE)


def predict(prompt, video_data, temperature):
    strategy = 'chat'

    video = load_video(video_data, strategy=strategy)

    history = []
    query = prompt
    inputs = model.build_conversation_input_ids(
        tokenizer=tokenizer,
        query=query,
        images=[video],
        history=history,
        template_version=strategy
    )
    inputs = {
        'input_ids': inputs['input_ids'].unsqueeze(0).to('cuda'),
        'token_type_ids': inputs['token_type_ids'].unsqueeze(0).to('cuda'),
        'attention_mask': inputs['attention_mask'].unsqueeze(0).to('cuda'),
        'images': [[inputs['images'][0].to('cuda').to(TORCH_TYPE)]],
    }
    gen_kwargs = {
        "max_new_tokens": 2048,
        "pad_token_id": 128002,
        "top_k": 1,
        "do_sample": False,
        "top_p": 0.1,
        "temperature": temperature,
    }
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
        outputs = outputs[:, inputs['input_ids'].shape[1]:]
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response


def test():
    prompt = "Please describe this video in detail."
    temperature = 0.1
    video_data = open('/home/dell/YX/FLUX-task/FlexiAct/Open-VFX/val/gt_videos/Levitate_it-3.mp4', 'rb').read()
    response = predict(prompt, video_data, temperature)
    print(response)


if __name__ == '__main__':
    test()





# import io
# import os
# import argparse
# import numpy as np
# import torch
# from decord import cpu, VideoReader, bridge
# from transformers import AutoModelForCausalLM, AutoTokenizer

# MODEL_PATH = "/home/dell/YX/FLUX-task/FlexiAct/cogvlm2-llama3-caption"

# DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# TORCH_TYPE = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[
#     0] >= 8 else torch.float16

# parser = argparse.ArgumentParser(description="CogVLM2-Video Batch Processing")
# parser.add_argument('--quant', type=int, choices=[4, 8], help='Enable 4-bit or 8-bit precision loading', default=0)
# parser.add_argument('--input_dir', type=str, required=True, help='Directory containing video files')
# parser.add_argument('--video_extensions', nargs='+', default=['.mp4', '.avi', '.mov', '.mkv'], 
#                     help='Video file extensions to process')
# args = parser.parse_args()


# def load_video(video_data, strategy='chat'):
#     bridge.set_bridge('torch')
#     mp4_stream = video_data
#     num_frames = 24
#     decord_vr = VideoReader(io.BytesIO(mp4_stream), ctx=cpu(0))

#     frame_id_list = None
#     total_frames = len(decord_vr)
#     if strategy == 'base':
#         clip_end_sec = 60
#         clip_start_sec = 0
#         start_frame = int(clip_start_sec * decord_vr.get_avg_fps())
#         end_frame = min(total_frames,
#                         int(clip_end_sec * decord_vr.get_avg_fps())) if clip_end_sec is not None else total_frames
#         frame_id_list = np.linspace(start_frame, end_frame - 1, num_frames, dtype=int)
#     elif strategy == 'chat':
#         timestamps = decord_vr.get_frame_timestamp(np.arange(total_frames))
#         timestamps = [i[0] for i in timestamps]
#         max_second = round(max(timestamps)) + 1
#         frame_id_list = []
#         for second in range(max_second):
#             closest_num = min(timestamps, key=lambda x: abs(x - second))
#             index = timestamps.index(closest_num)
#             frame_id_list.append(index)
#             if len(frame_id_list) >= num_frames:
#                 break

#     video_data = decord_vr.get_batch(frame_id_list)
#     video_data = video_data.permute(3, 0, 1, 2)
#     return video_data


# tokenizer = AutoTokenizer.from_pretrained(
#     MODEL_PATH,
#     trust_remote_code=True,
# )

# model = AutoModelForCausalLM.from_pretrained(
#     MODEL_PATH,
#     torch_dtype=TORCH_TYPE,
#     trust_remote_code=True
# ).eval().to(DEVICE)


# def predict(prompt, video_data, temperature):
#     strategy = 'chat'

#     video = load_video(video_data, strategy=strategy)

#     history = []
#     query = prompt
#     inputs = model.build_conversation_input_ids(
#         tokenizer=tokenizer,
#         query=query,
#         images=[video],
#         history=history,
#         template_version=strategy
#     )
#     inputs = {
#         'input_ids': inputs['input_ids'].unsqueeze(0).to('cuda'),
#         'token_type_ids': inputs['token_type_ids'].unsqueeze(0).to('cuda'),
#         'attention_mask': inputs['attention_mask'].unsqueeze(0).to('cuda'),
#         'images': [[inputs['images'][0].to('cuda').to(TORCH_TYPE)]],
#     }
#     gen_kwargs = {
#         "max_new_tokens": 2048,
#         "pad_token_id": 128002,
#         "top_k": 1,
#         "do_sample": False,
#         "top_p": 0.1,
#         "temperature": temperature,
#     }
#     with torch.no_grad():
#         outputs = model.generate(**inputs,** gen_kwargs)
#         outputs = outputs[:, inputs['input_ids'].shape[1]:]
#         response = tokenizer.decode(outputs[0], skip_special_tokens=True)
#         return response


# def process_video_file(video_path, video_index, prompt, temperature):
#     """处理单个视频文件并打印结果"""
#     try:
#         # 读取视频数据
#         with open(video_path, 'rb') as f:
#             video_data = f.read()
        
#         # 处理视频
#         print(f"正在处理视频 {video_index}: {video_path}")
#         response = predict(prompt, video_data, temperature)
        
#         # 按要求格式打印结果
#         print(f"\n视频{video_index}：{response}\n")
#         print("-" * 100)  # 分隔线，方便区分不同视频的输出
#         return True
    
#     except Exception as e:
#         print(f"\n处理视频{video_index}时出错 ({video_path}): {str(e)}\n")
#         print("-" * 100)
#         return False


# def process_video_folder(input_dir, video_extensions, prompt, temperature):
#     """处理文件夹中所有视频文件并按顺序打印结果"""
#     # 先收集所有视频文件路径
#     video_files = []
#     for root, dirs, files in os.walk(input_dir):
#         for file in files:
#             # 检查文件扩展名是否符合要求
#             if any(file.lower().endswith(ext) for ext in video_extensions):
#                 video_path = os.path.join(root, file)
#                 video_files.append(video_path)
    
#     # 按顺序处理所有视频
#     processed_count = 0
#     for i, video_path in enumerate(video_files, 1):  # 从1开始计数
#         if process_video_file(video_path, i, prompt, temperature):
#             processed_count += 1
    
#     print(f"\n处理完成。共处理 {processed_count}/{len(video_files)} 个视频。")


# def main():
#     prompt = "Scene of cutting a cake, Please describe this video in detail."
#     temperature = 0.1
    
#     # 处理文件夹中的所有视频
#     process_video_folder(
#         input_dir=args.input_dir,
#         video_extensions=args.video_extensions,
#         prompt=prompt,
#         temperature=temperature
#     )


# if __name__ == '__main__':
#     main()
    
# 注意导入方式的变化
from modelscope.hub.snapshot_download import snapshot_download

# 下载模型到指定目录
model_dir = snapshot_download(
    model_id="ZhipuAI/cogvlm2-llama3-caption",
    local_dir="/home/dell/YX/FLUX-task/FlexiAct/cogvlm2-llama3-caption",  # 本地保存路径
)
print(f"模型已下载至：{model_dir}")
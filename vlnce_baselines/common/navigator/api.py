'''api.py'''

from openai import OpenAI
import torch
import numpy as np

import sys
import os

from tenacity import retry, wait_random_exponential, stop_after_attempt

import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings

from transformers import AutoConfig, AutoModelForCausalLM
from SpatialBot3B.configuration_bunny_phi import *
from SpatialBot3B.modeling_bunny_phi import *

AutoConfig.register("bunny-phi", BunnyPhiConfig)
AutoModelForCausalLM.register(BunnyPhiConfig, BunnyPhiForCausalLM)
transformers.logging.set_verbosity_error()
transformers.logging.disable_progress_bar()
warnings.filterwarnings('ignore')

from recognize_anything.ram.models import ram
from recognize_anything.ram import inference_ram
from recognize_anything.ram import get_transform



from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

@dataclass
class Subtask:
    """
    Subtask类，自定义的数据结构，组成instruction queue
    Represents a parsed subtask from the instruction.
    Fields can be None if not applicable or specified.
    """
    action: Optional[str] = None
    direction: Optional[str] = None
    preposition: Optional[str] = None
    landmark: Optional[str] = None
    
    def __str__(self):
        """方便打印和日志记录"""
        a = self.action if self.action is not None else 'None'
        d = self.direction if self.direction is not None else 'None'
        p = self.preposition if self.preposition is not None else 'None'
        l = self.landmark if self.landmark is not None else 'None'
        return f"[action={a}, dir={d}, prep={p}, land={l}]"

    def to_dict(self):
        """转换为字典，方便序列化或传递给LLM提示词"""
        return {
            "action": self.action,
            "direction": self.direction,
            "preposition": self.preposition,
            "landmark": self.landmark
        }


@dataclass
class TrajectoryTreeNode:
    """
    Represents a node in the trajectory tree.
    """
    viewpoint_id: int                    # 航点ID (e.g., "0", "1")
    observation: str                     # 在该航点的观察描述 (来自 observe_environment)
    thought: str                         # 选择该航点时的思考/理由 (来自 test_decisions)
    subtask_at_time: Optional['Subtask'] # 选择该航点时的当前子任务
    parent: Optional['TrajectoryTreeNode'] = None  # 父节点
    children: List['TrajectoryTreeNode'] = field(default_factory=list) # 子节点列表
    # 可以添加更多属性，如时间戳、坐标估计等
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_child(self, child_node: 'TrajectoryTreeNode'):
        """添加一个子节点"""
        child_node.parent = self
        self.children.append(child_node)

    def get_path_from_root(self) -> List['TrajectoryTreeNode']:
        """获取从根节点到当前节点的路径"""
        path = []
        current = self
        while current is not None:
            path.append(current)
            current = current.parent
        return path[::-1] # Reverse to get root-to-current order

    def get_recent_path(self, depth: int) -> List['TrajectoryTreeNode']:
        """获取从当前节点向上追溯指定深度的路径"""
        path = []
        current = self
        for _ in range(depth):
            if current is None:
                break
            path.append(current)
            current = current.parent
        return path[::-1] # Reverse to get ancestor-to-current order





class llmClient:
    def __init__(self, model_type = '', api_key=None, base_url=None):
        '''
        Initialize LLM client based on model type and API key.
        
        Args:
            model_type (str): Either "gpt" or "opensource"
            api_key (str): API key for OpenAI (if using GPT)
        '''
        # Configure based on model type
        if model_type == "gpt-4o-2024-08-06":
            self.model = model_type
            self.client = OpenAI(api_key=api_key)
            
        elif model_type == "Qwen/Qwen2-72B":
            self.model = model_type
            self.client = OpenAI(
                api_key="not-needed",  # This value doesn't matter for local deployment
                base_url="http://0.0.0.0:23333/v1"
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}. Use 'gpt' or 'opensource'.")
        
        print(f"Initialized LLM client with model: {self.model}")

    def set_model(self, model):
        self.model = model

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def _completion_with_backoff(self, **kwargs):
        return self.client.chat.completions.create(**kwargs)

    def gpt_infer(self, system_prompt, user_prompt, num_output=1):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        request_params = {
            "model": self.model,
            "messages": messages,
            "temperature": 0
        }
        
        if num_output == 1:
            chat_response = self._completion_with_backoff(**request_params)
            answer = chat_response.choices[0].message.content
            return answer
        else:
            responses = []
            for _ in range(num_output):
                chat_response = self._completion_with_backoff(**request_params)
                responses.append(chat_response.choices[0].message.content)
            return responses

    
class spatialClient:
    """
    场景感知客户端类，负责加载和调用 Recognize Anything Model (RAM) 和 SpatialBot。
    这是 Open-Nav 中 "Scene Perception" 部分的核心实现。
    """
    def __init__(self, device):
        self.device = device
        self.ram_path = "./recognize_anything/pretrained/ram_swin_large_14m.pth"
        self.spatialbot_path = "./SpatialBot3B"
        view_record_path = "cache_files/view_cache.json"
        try:
            self.spatialbot_model = AutoModelForCausalLM.from_pretrained(
                self.spatialbot_path,
                torch_dtype=torch.float16, # float32 for cpu
                device_map='auto',
                trust_remote_code=True)
            self.spatialbot_tokenizer = AutoTokenizer.from_pretrained(
                self.spatialbot_path,
                trust_remote_code=True)
            
            self.ram_transform = get_transform(image_size=224) 
            self.ram_model = ram(pretrained=self.ram_path, image_size=224, vit='swin_l').eval().to(self.device)
        except Exception as e:
            print(f"Error in loading RAM or SpatialBot: {e}")
            
    def ram_img_tagging(self, image):
        """
        使用 RAM 模型对单张图像进行标签预测（物体识别）。

        Args:
            image (PIL.Image): 输入的 RGB 图像。

        Returns:
            str: 由 RAM 模型识别出的物体标签，以逗号分隔的字符串。
                 例如: "chair, table, window, lamp"
        """
        ram_img = self.ram_transform(image).unsqueeze(0).to(self.device)
        img_tags = inference_ram(ram_img, self.ram_model)[0]
        return img_tags
    
    def spatialbot_description(self, image_dict, prompt):
        """
        使用 SpatialBot 模型生成对图像的详细空间描述。

        Args:
            image_dict (dict): 包含 'rgb' 和 'depth' 图像的字典。
                               - 'rgb': PIL.Image 对象 (RGB 图像)
                               - 'depth': PIL.Image 对象 (深度图)
            prompt (str): 给 SpatialBot 的指令

        Returns:
            str: SpatialBot 生成的文本描述。
        """
        # 1. 构造输入给 SpatialBot 的文本提示
        offset_bos = 0
        # 构造一个标准的对话格式，其中包含图像占位符 <image 1> 和 <image 2>
        # TODO：这text是干啥
        text = f"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: <image 1>\n<image 2>\n{prompt} ASSISTANT:"
        
        # 2. 对文本进行分词处理
        #    - 将包含图像占位符的文本按占位符分割
        #    - 对每个分割后的文本块分别进行分词
        text_chunks = [self.spatialbot_tokenizer(chunk).input_ids for chunk in text.split('<image 1>\n<image 2>\n')]
        
        # 3. 构造最终的输入 ID 张量
        #    - text_chunks[0]: 第一个文本块的 token ID
        #    - [-201]: 图像1的特殊占位符 ID (模型自定义)
        #    - [-202]: 图像2的特殊占位符 ID (模型自定义)
        #    - text_chunks[1][offset_bos:]: 第二个文本块的 token ID (从 offset_bos 开始)
        #    - torch.tensor(...).unsqueeze(0): 转换为张量并增加批次维度
        #    - .to(self.device): 移动到指定设备
        input_ids = torch.tensor(text_chunks[0] + [-201] + [-202] + text_chunks[1][offset_bos:], dtype=torch.long).unsqueeze(0).to(self.device)
        
        # 4. 准备图像输入
        image1 = image_dict['rgb']
        image2 = image_dict['depth']

        # 5. 处理深度图格式
        #    检查深度图的通道数
        channels = len(image2.getbands())
        if channels == 1:   # 如果是单通道深度图
            # 将单通道深度图转换为三通道 RGB 格式
            # 这是一种常见的可视化深度图的方法
            img = np.array(image2)
            height, width = img.shape
            three_channel_array = np.zeros((height, width, 3), dtype=np.uint8)
            three_channel_array[:, :, 0] = (img // 1024) * 4    # R
            three_channel_array[:, :, 1] = (img // 32) * 8  # G
            three_channel_array[:, :, 2] = (img % 32) * 8   # B
            image2 = Image.fromarray(three_channel_array, 'RGB')    # 转换回 PIL Image
        image_tensor = self.spatialbot_model.process_images([image1,image2], self.spatialbot_model.config).to(dtype=self.spatialbot_model.dtype, device=self.device)
        
        # 7. 确保视觉塔（Vision Tower）在 GPU 上
        self.spatialbot_model.get_vision_tower().to('cuda')

        # 8. 调用模型的 generate 方法生成文本
        #    - input_ids: 文本输入 ID
        #    - images=image_tensor: 图像输入张量
        #    - max_new_tokens=200: 最多生成 200 个新 token
        #    - use_cache=True: 使用缓存加速生成
        #    - repetition_penalty=1.0: 重复惩罚因子
        #    - [0]: 取第一个（也是唯一一个）批次的输出
        output_ids = self.spatialbot_model.generate(
            input_ids,
            images=image_tensor,
            max_new_tokens=200, 
            use_cache=True,
            repetition_penalty=1.0 
        )[0]

        # 9. 解码生成的 token ID 为文本
        #    - output_ids[input_ids.shape[1]:]: 只取新生成的部分（去掉输入部分）
        #    - skip_special_tokens=True: 跳过特殊 token
        #    - .strip(): 去除首尾空白字符
        return self.spatialbot_tokenizer.decode(output_ids[input_ids.shape[1]:], skip_special_tokens=True).strip()
    
    def observe_view(self, logger, current_step, direction_idx, direction_image):
        """
        [Scene Perception 的核心实现]
        对一个特定方向（航点）的视图进行完整的场景观察。
        这是 `spatialNavigator.py` 中 `observe_environment` 方法调用的函数。

        Args:
            logger: 日志记录器。
            current_step (int): 当前导航步数。
            direction_idx (str): 当前方向（航点）的索引。
            direction_image (dict): 包含该方向 'rgb' 和 'depth' 图像的字典。

        Returns:
            str: 格式化后的完整观察结果字符串，包含物体标签和空间描述。
                 例如: "Direction 2 Direction Viewpoint ID: 2 in Step ID: 1 Elevation: Eye Level Scene Description: There is a chair about 2 meters away... Scene Objects: chair, table, window;"
        """
        # 1. [调用 RAM] 获取图像中的物体标签
        #    调用 ram_img_tagging 方法处理 RGB 图像
        img_tags = self.ram_img_tagging(direction_image['rgb'])

        # 2. [调用 SpatialBot] 获取详细的空间描述
        #    定义给 SpatialBot 的提示词，要求它描述物体和距离

        # TODO：修改prompt

        spatial_scene_description_prompt = "What objects are in the image, and how far are these objects from the camera, calculate the result in meter."
        #    调用 spatialbot_description 方法处理 RGB 和深度图
        spatial_scene_description = self.spatialbot_description(direction_image, spatial_scene_description_prompt)

        # TODO：我觉得其实根本不需要这个RAM来识别物体呀，spacialbot一样得去识别。要是两个识别的不一样怎么办。

        # 3. [融合信息] 将 RAM 和 SpatialBot 的输出融合成一个描述
        view_observation = f"Scene Description: {spatial_scene_description} Scene Objects: {img_tags}; "

        # 4. [格式化输出] 添加方向、步数、视角高度等元信息
        observe_result = f"Direction {direction_idx} Direction Viewpoint ID: {direction_idx} in Step ID: {current_step} Elevation: Eye Level "  + view_observation
        
        return observe_result
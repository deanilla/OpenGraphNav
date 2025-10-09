'''vlnce_baselines/common/navigator/api.py'''

import re
import time
from openai import OpenAI
import torch
import numpy as np

import sys
import os
import warnings

from tenacity import retry, wait_random_exponential, stop_after_attempt

import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

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

from vlnce_baselines.common.graph.scene_graph import *

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
import requests
from io import BytesIO


@dataclass
class Subtask:
    """
    Subtask类，自定义的数据结构，组成instruction queue
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
    trajectory tree中的节点类实现
    """
    viewpoint_id: int                    # 航点ID (e.g., "0", "1")
    observation: str                     # TODO：原为在该航点的观察描述 (来自 observe_environment)，须删去，功能由scene graph中waypoint node的子图替代。
    thought: str                         # HACK：选择该航点时的思考/理由 (来自 test_decisions)，用于CoT
    subtask_at_time: Optional['Subtask']        # 选择该航点时的子任务
    parent: Optional['TrajectoryTreeNode'] = None  # 父节点
    children: List['TrajectoryTreeNode'] = field(default_factory=list) # 子节点列表
    # TODO：不需要添加更多属性，如时间戳、坐标估计等
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_child(self, child_node: 'TrajectoryTreeNode'):
        """添加一个子节点"""
        child_node.parent = self
        self.children.append(child_node)

    # TODO：这两个函数为回溯/剪枝设计，暂时未用
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


@dataclass
class Snapshot:
    """
    快照类，用于记录特定时刻的任务状态与环境信息
    """
    step_id: int                           # 步骤ID
    viewpoint_id: str                      # 当前航点ID
    current_subtask: Optional[Subtask]     # 当前执行的子任务
    completed_subtasks: List[Subtask]      # 已完成的子任务列表
    environment_description: str           # 环境简要描述
    action_executed: str                   # 刚刚执行的动作
    timestamp: float = field(default_factory=time.time)  # 时间戳
    
    def __str__(self):
        return f"Snapshot(step={self.step_id}, vp={self.viewpoint_id}, task={self.current_subtask})"
    
    def to_dict(self):
        """转换为字典，便于序列化"""
        return {
            "step_id": self.step_id,
            "viewpoint_id": self.viewpoint_id,
            "current_subtask": self.current_subtask.to_dict() if self.current_subtask else None,
            "completed_subtasks": [task.to_dict() for task in self.completed_subtasks],
            "environment_description": self.environment_description,
            "action_executed": self.action_executed,
            "timestamp": self.timestamp
        }


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
    场景感知客户端类，负责加载和调用 VLM。
    Open-Nav 中 "Scene Perception" 部分的核心实现。
    """
    def __init__(self, device):
        self.device = device
        self.tag_model_id = "remyxai/SpaceOm"
        self.spacial_model_id = "remyxai/SpaceThinker-Qwen2.5VL-3B"
        view_record_path = "cache_files/view_cache.json"    # FIXME

    def vlm_infer(
            self, image, prompt, 
            system_message = (
                "You are VL-Thinking 🤔, a helpful assistant with excellent reasoning ability. "
                "You should first think about the reasoning process and then provide the answer. "
                "Use <think>...</think> and <answer>...</answer> tags."
            ), 
            num_output=1
        ):

        # Load model and processor
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_id, device_map="auto", torch_dtype=torch.bfloat16
        )
        processor = AutoProcessor.from_pretrained(self.model_id)

        # Preprocess image
        if image.width > 512:
            ratio = image.height / image.width
            image = image.resize((512, int(512 * ratio)), Image.Resampling.LANCZOS)

        # Format input
        chat = [
            {"role": "system", "content": [{"type": "text", "text": system_message}]},
            {"role": "user", "content": [{"type": "image", "image": image},
                                        {"type": "text", "text": prompt}]}
        ]
        text_input = processor.apply_chat_template(chat, tokenize=False,
                                                        add_generation_prompt=True)

        # Tokenize
        inputs = processor(text=[text_input], images=[image],
                                            return_tensors="pt").to("cuda")

        # Generate response
        generated_ids = model.generate(**inputs, max_new_tokens=1024)
        output = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        return output
    
    def spacial_vlm_infer(
            self, image1, image2, prompt, 
            system_message = (
                "You are VL-Thinking 🤔, a helpful assistant with excellent reasoning ability. "
                "You should first think about the reasoning process and then provide the answer. "
                "Use <think>...</think> and <answer>...</answer> tags."
            ), 
            num_output=1
        ):
        """
        使用支持双图像输入的VLM进行推理
        
        Args:
            image1: 第一张图像（如RGB图像）
            image2: 第二张图像（如深度图）
            prompt: 用户提示
            system_message: 系统提示
            num_output: 输出数量
        """
        # Load model and processor
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_id, device_map="auto", torch_dtype=torch.bfloat16
        )
        processor = AutoProcessor.from_pretrained(self.model_id)

        # Preprocess images
        for img in [image1, image2]:
            if img.width > 512:
                ratio = img.height / img.width
                img = img.resize((512, int(512 * ratio)), Image.Resampling.LANCZOS)

        # Format input - include both images
        chat = [
            {"role": "system", "content": [{"type": "text", "text": system_message}]},
            {"role": "user", "content": [
                {"type": "image", "image": image1},
                {"type": "image", "image": image2},
                {"type": "text", "text": prompt}
            ]}
        ]
        text_input = processor.apply_chat_template(chat, tokenize=False,
                                                        add_generation_prompt=True)

        # Tokenize - pass both images
        inputs = processor(text=[text_input], 
                        images=[image1, image2],
                        return_tensors="pt").to("cuda")

        # Generate response
        generated_ids = model.generate(**inputs, max_new_tokens=1024)
        output = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        return output
            
    def img_tagging(self, image):
        """
        使用 VLM 对单张图像进行标签预测（物体识别）。

        Args:
            image (PIL.Image): 输入的 RGB 图像。

        Returns:
            str: 由 VLM 识别出的物体标签，以逗号分隔的字符串。
                 例如: "chair, table, window, lamp"
        """
        img_tagging_prompt = "What objects can you see in the image?"
        img_tagging_system_prompt = (
            "You are VL-Thinking 🤔, a helpful object detection agent with excellent reasoning ability. "
            "You should first think about the reasoning process and then provide the answer. "
            "Use <think>...</think> and <answer>...</answer> tags."
            "The answer in <answer>...</answer> tags must be strings of object names seperated by comma."
        )
        output = self.vlm_infer(image, img_tagging_prompt, img_tagging_system_prompt)
        
        # 从字符串中提取 <answer>...</answer> 标签中的内容
        answer_match = re.search(r'<answer>(.*?)</answer>', output, re.DOTALL)
        img_tags = answer_match.group(1).strip() if answer_match else output

        return img_tags
    
    def spatial_description(self, image_dict, prompt):
        """
        使用 SpatialBot 模型生成对图像的描述或结构化信息。

        Args:
            image_dict (dict): 包含 'rgb' 和 'depth' 图像的字典。
            prompt (str): 给 SpatialBot 的指令。

        Returns:
            str: SpatialBot 生成的文本描述。
        """
       
        # 提取图像输入
        image1 = image_dict['rgb']
        image2 = image_dict['depth']

        channels = len(image2.getbands())
        if channels == 1:   # 如果是单通道深度图
            # 将单通道深度图转换为三通道 RGB 格式
            img = np.array(image2)
            height, width = img.shape
            three_channel_array = np.zeros((height, width, 3), dtype=np.uint8)
            three_channel_array[:, :, 0] = (img // 1024) * 4    # R
            three_channel_array[:, :, 1] = (img // 32) * 8      # G
            three_channel_array[:, :, 2] = (img % 32) * 8       # B
            image2 = Image.fromarray(three_channel_array, 'RGB')

        spatial_description_prompt = prompt
        spatial_description_system_prompt = (
            "You are VL-Thinking 🤔, a helpful spacial scene observation agent with excellent reasoning ability. "
            "You should first think about the reasoning process and then provide the answer. "
            "Use <think>...</think> and <answer>...</answer> tags."
        )
        output = self.spacial_vlm_infer(image1, image2, spatial_description_prompt, spatial_description_system_prompt)
        
        # 从字符串中提取 <answer>...</answer> 标签中的内容
        answer_match = re.search(r'<answer>(.*?)</answer>', output, re.DOTALL)
        spatial_description = answer_match.group(1).strip() if answer_match else output
                
        return spatial_description


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
        """

        img_tags = self.img_tagging(direction_image['rgb'])

        spatial_scene_description_prompt = "What objects are in the image, and how far are these objects from the camera, calculate the result in meter."
        spatial_scene_description = self.spatial_description(direction_image, spatial_scene_description_prompt)

        # 将以上两个输出融合成一个描述
        view_observation = f"Scene Description: {spatial_scene_description} Scene Objects: {img_tags}; "

        # 添加方向、步数、视角高度等元信息
        observe_result = f"Direction {direction_idx} Direction Viewpoint ID: {direction_idx} in Step ID: {current_step} Elevation: Eye Level "  + view_observation
        
        return observe_result
        
    # 新增函數
    def update_scene_graph_from_observation(
        self, 
        logger, 
        current_waypoint_id: str, 
        direction_image: dict, 
        current_subgraph: Optional[SceneGraph] = None
    ) -> Tuple[List[SceneNode], List[SceneEdge]]:
        """
        在智能体到达新航点后，基于当前观察更新 Scene Graph 的局部区域。

        Args:
            logger: 日志记录器。
            current_waypoint_id (str): 智能体当前所在的航点 ID。
            direction_image (dict): 包含该航点 'rgb' 和 'depth' 图像的字典。
            current_subgraph (Optional[SceneGraph]):
                当前以 current_waypoint_id 为中心的 SceneGraph 子图。

        Returns:
            Tuple[List[SceneNode], List[SceneEdge]]: 
            一个元组，包含两部分：
            1. List[SceneNode]: 基于当前观察，需要更新或添加到 Scene Graph 中的节点列表。
            2. List[SceneEdge]: 基于当前观察，需要更新或添加到 Scene Graph 中的边列表。
            这些节点和边将由调用者负责整合到主 SceneGraph 中。
        """
        log_prefix = f"[update_scene_graph_from_obs|WP:{current_waypoint_id}]"
        logger.info(f"{log_prefix} Starting scene graph update from observation.")

        new_nodes: List[SceneNode] = []
        new_edges: List[SceneEdge] = []

        # 1. 确保当前航点节点存在于图中（如果图是空的或不包含该节点）
        # 注意：ID 格式需要与 initialize_scene_graph 中保持一致
        current_wp_node_id = f"wp_{current_waypoint_id}"
        # 不直接创建节点，而是返回它，让调用者决定是否需要添加。如果子图已提供，检查它是否已存在
        wp_node_exists_in_subgraph = False
        if current_subgraph:
            wp_node_exists_in_subgraph = current_subgraph.graph.has_node(current_wp_node_id)

        if not wp_node_exists_in_subgraph:
            logger.info(f"{log_prefix} Current waypoint node '{current_wp_node_id}' not found in subgraph or subgraph is None. Will include it in updates if needed.")
            # 创建航点节点（如果需要添加到主图）
            wp_node = SceneNode(
                id=current_wp_node_id,
                type=NodeType.WAYPOINT,
                attributes={
                    'viewpoint_id': int(current_waypoint_id) if current_waypoint_id.isdigit() else current_waypoint_id,
                }
            )
            new_nodes.append(wp_node)
        else:
            logger.info(f"{log_prefix} Current waypoint node '{current_wp_node_id}' found in subgraph.")

        # TODO: 调用 VLM 获取结构化的物体及其关系
        # TODO：优化此prompt
        spatial_scene_prompt = (
            "Analyze the image from an agent's viewpoint inside a room. "
            "Your task is twofold: "
            "1.  List Objects: Identify distinct, prominent objects. For each, provide 'name' and 'id' (a unique identifier for this object in this scene, e.g., 'sofa_1', 'chair_A'). "
            "2.  List Spatial Relationships: Describe spatial relationships between the objects and between the objects and the agent's viewpoint. "
            "Use clear, concise terms like 'left_of', 'right_of', 'in_front_of', 'behind', 'near', 'far', 'inside' (for parts), 'part_of' (for parts). "
            "Format your response strictly as a JSON object with two keys: 'objects' (an array of object dicts) and 'relationships' (an array of relationship dicts). "
            "Each relationship dict must have 'source_id', 'relation' (the spatial term), and 'target_id'. "
            "The agent's viewpoint can be referred to by its ID, which is provided later. "
            "Example Output: "
            "{"
            '  "objects": ['
            '    {"id": "sofa_1", "name": "sofa"},'
            '    {"id": "lamp_1", "name": "lamp"}'
            '  ],'
            '  "relationships": ['
            '    {"source_id": "sofa_1", "relation": "left_of", "target_id": "lamp_1"},'
            '    {"source_id": "lamp_1", "relation": "right_of", "target_id": "sofa_1"}'
            '  ]'
            "}"
            "Now, provide the JSON output for the image."
            # TODO：在 Prompt 中或通过其他方式告诉模型当前航点的 ID ---
        )
        
        # 动态修改 Prompt 以包含航点 ID
        spatial_scene_prompt_with_wp = f"{spatial_scene_prompt}\n\nThe agent's current viewpoint ID is: {current_wp_node_id}"

        try:
            logger.info(f"{log_prefix} Calling VLM for structured objects and relationships...")
            # 调用 VLM 获取输出
            vlm_output = self.spatial_description(
                direction_image, 
                spatial_scene_prompt_with_wp, 
            )
            logger.info(f"{log_prefix} VLM output type: {type(vlm_output)}")
            logger.debug(f"{log_prefix} VLM raw output: {vlm_output}")
        except Exception as e:
            logger.error(f"{log_prefix} Error calling VLM: {e}")
            vlm_output = None

        # 4. 处理 VLM 的结构化输出
        detected_objects = []
        detected_relationships = []
        if isinstance(vlm_output, dict):
            detected_objects = vlm_output.get("objects", [])
            detected_relationships = vlm_output.get("relationships", [])
            logger.info(f"{log_prefix} VLM found {len(detected_objects)} objects and {len(detected_relationships)} relationships.")
        else:
            logger.warning(f"{log_prefix} VLM did not return a valid dict. Output was: {vlm_output}")

        # 5. 将检测到的对象转换为 SceneNode
        # 使用一个字典来跟踪新创建的节点 ID，避免重复处理
        processed_object_ids = set()
        for obj_dict in detected_objects:
            if isinstance(obj_dict, dict) and 'id' in obj_dict and 'name' in obj_dict:
                obj_id = obj_dict['id']
                obj_name = obj_dict['name']
                # TODO：为了避免 ID 冲突，可以考虑加上航点前缀
                full_obj_id = obj_id 
                
                if full_obj_id in processed_object_ids:
                    continue

                # 创建 Object 节点
                obj_node = SceneNode(
                    id=full_obj_id,
                    type=NodeType.OBJECT,
                    attributes={
                        'category': obj_name,
                        'detected_by': 'VLM',
                    }
                )
                new_nodes.append(obj_node)
                processed_object_ids.add(full_obj_id)
                logger.debug(f"{log_prefix} Created Object node: {obj_node}")

        # 6. 将检测到的关系转换为 SceneEdge
        # 同时处理物体与航点的关系，以及物体之间的关系
        for rel_dict in detected_relationships:
            if isinstance(rel_dict, dict) and all(k in rel_dict for k in ('source_id', 'relation', 'target_id')):
                src_id = rel_dict['source_id']
                relation = rel_dict['relation']
                tgt_id = rel_dict['target_id']
                
                # 确定边的类型
                # TODO：边分类判断逻辑
                edge_type = EdgeType.SPATIAL # 默认是空间关系
                if relation in ['inside', 'part_of']:
                    edge_type = EdgeType.AFFILIATION
                
                # 创建边
                # TODO：如果关系涉及未在此处处理的节点（例如，子图中已存在的其他物体），主图更新逻辑需要处理
                edge = SceneEdge(
                    source_id=src_id,
                    target_id=tgt_id,
                    relation=relation,
                    type=edge_type,
                    confidence=0.9 # TODO：假设的置信度，可以基于 LLM 的确定性或其他因素调整
                )
                new_edges.append(edge)
                logger.debug(f"{log_prefix} Created Edge: {edge}")

        logger.info(f"{log_prefix} Scene graph update preparation complete. "
                    f"Returning {len(new_nodes)} new/updated nodes and {len(new_edges)} new/updated edges.")
        
        return new_nodes, new_edges

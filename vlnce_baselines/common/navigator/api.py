'''vlnce_baselines/common/navigator/api.py'''

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
        view_record_path = "cache_files/view_cache.json"    # FIXME：未使用
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
    
    def spatialbot_description(self, image_dict, prompt, force_json=True):
        """
        使用 SpatialBot 模型生成对图像的描述或结构化信息。

        Args:
            image_dict (dict): 包含 'rgb' 和 'depth' 图像的字典。
            prompt (str): 给 SpatialBot 的指令。
            force_json (bool, optional): 是否强制模型输出 JSON 格式。默认为 False。

        Returns:
            str or dict: SpatialBot 生成的文本描述，或解析后的 JSON 字典。
        """
        # --- 根据 force_json 调整 Prompt ---
        if force_json:
            # 强制 JSON 输出的提示词
            json_instruction = (
                "Your response must be a valid JSON object. "
                "Do not include any other text, code blocks, or markdown. "
                "Just output the raw JSON."
            )
            prompt = f"{prompt}\n{json_instruction}"
        # 1. 构造输入给 SpatialBot 的文本提示
        offset_bos = 0
        # TODO：这段text是干啥的？
        text = f"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: <image 1>\n<image 2>\n{prompt} ASSISTANT:"
        
        # 2. 对文本进行分词处理
        text_chunks = [self.spatialbot_tokenizer(chunk).input_ids for chunk in text.split('<image 1>\n<image 2>\n')]
        
        # 3. 构造最终的输入 ID 张量
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
            image2 = Image.fromarray(three_channel_array, 'RGB')    # FIXME：Open-Nav原版就没有导入，不知道为什么
        image_tensor = self.spatialbot_model.process_images([image1,image2], self.spatialbot_model.config).to(dtype=self.spatialbot_model.dtype, device=self.device)
        
        # 7. 确保视觉塔（Vision Tower）在 GPU 上
        self.spatialbot_model.get_vision_tower().to('cuda')

        # 8. 调用模型的 generate 方法生成文本
        # --- 为 JSON 输出调整生成参数 ---
        generate_kwargs = {
            "input_ids": input_ids,
            "images": image_tensor,
            "max_new_tokens": 300 if force_json else 200, # JSON 可能需要更多 token
            "use_cache": True,
            "repetition_penalty": 1.0
        }
        # 如果强制 JSON，可以尝试降低 temperature 以获得更确定的输出
        if force_json:
            generate_kwargs["temperature"] = 0.0
            generate_kwargs["do_sample"] = False
            
        output_ids = self.spatialbot_model.generate(**generate_kwargs)[0]

        # 9. 解码生成的 token ID 为文本
        response_text = self.spatialbot_tokenizer.decode(output_ids[input_ids.shape[1]:], skip_special_tokens=True).strip()
        
        # --- 修改点3: 尝试解析 JSON ---
        if force_json:
            try:
                # 简单清理，尝试提取可能被包裹的 ```json ... ``` 或其他符号
                import json
                cleaned_text = response_text.strip()
                if cleaned_text.startswith("```json"):
                    cleaned_text = cleaned_text[7:]
                if cleaned_text.endswith("```"):
                    cleaned_text = cleaned_text[:-3]
                cleaned_text = cleaned_text.strip()
                
                parsed_json = json.loads(cleaned_text)
                print(f"SpatialBot returned valid JSON: {parsed_json}") # 调试信息
                return parsed_json
            except (json.JSONDecodeError, Exception) as e:
                print(f"Warning: force_json=True, but failed to parse SpatialBot output as JSON: {e}")
                print(f"Raw output was: {response_text}")
                # 解析失败则返回原始文本
                return response_text 
        else:
            return response_text

    # HACK：deprecated
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
        # 我们不直接创建节点，而是返回它，让调用者决定是否需要添加
        # 如果子图已提供，检查它是否已存在
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

        # 2. 调用 RAM 获取基础物体标签
        try:
            img_tags_str = self.ram_img_tagging(direction_image['rgb'])
            img_tags_list = [tag.strip() for tag in img_tags_str.split(',') if tag.strip()]
            logger.info(f"{log_prefix} RAM detected objects: {img_tags_list}")
        except Exception as e:
            logger.error(f"{log_prefix} Error calling RAM: {e}")
            img_tags_list = []

        # 3. 调用 SpatialBot 获取结构化的物体及其关系
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
            logger.info(f"{log_prefix} Calling SpatialBot for structured objects and relationships...")
            # 调用 SpatialBot 并强制尝试获取 JSON 格式输出
            spatialbot_output = self.spatialbot_description(
                direction_image, 
                spatial_scene_prompt_with_wp, 
                force_json=True
            )
            logger.info(f"{log_prefix} SpatialBot output type: {type(spatialbot_output)}")
            logger.debug(f"{log_prefix} SpatialBot raw output: {spatialbot_output}")
        except Exception as e:
            logger.error(f"{log_prefix} Error calling SpatialBot: {e}")
            spatialbot_output = None

        # 4. 处理 SpatialBot 的结构化输出
        detected_objects = []
        detected_relationships = []
        if isinstance(spatialbot_output, dict):
            detected_objects = spatialbot_output.get("objects", [])
            detected_relationships = spatialbot_output.get("relationships", [])
            logger.info(f"{log_prefix} SpatialBot found {len(detected_objects)} objects and {len(detected_relationships)} relationships.")
        else:
            logger.warning(f"{log_prefix} SpatialBot did not return a valid dict. Output was: {spatialbot_output}")

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
                        'detected_by': 'SpatialBot',
                        # 可以添加更多属性，如果 SpatialBot 提供了
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

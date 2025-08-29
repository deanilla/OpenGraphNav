'''vlnce_baselines/common/navigator/spatialNavigator.py'''
import re   # 正则表达式
import random
import json # get_subtasks中解析用
from vlnce_baselines.common.navigator.api import *
from vlnce_baselines.common.navigator.prompts import *

from typing import List, Dict, Optional, Tuple # 添加类型注解所需

from vlnce_baselines.common.graph.scene_graph import *


class Open_Nav():
    def __init__(self, device, llm_type, api_key):
        self.device = device
        self.llm = llmClient(llm_type, api_key)
        self.spatial = spatialClient(self.device)

        # 新增：初始化用于管理 subtask_queue 和 trajectory_tree 的状态变量
        self.current_subtask_queue: List[Subtask] = []
        self.trajectory_tree_root: Optional[TrajectoryTreeNode] = None
        self.current_trajectory_node: Optional[TrajectoryTreeNode] = None
        self.candidates_from_last_step: Dict[str, str] = {}

        # --- 新增：Scene Graph 管理 ---
        self.scene_graph: Optional[SceneGraph] = None
        
    # =====================================
    # ===== Instruction Comprehension =====
    # =====================================
    def get_subtasks(self, instruction: str) -> list[Subtask]:
        """
        将自然语言指令解析为 Subtask 对象的列表。
        """
        llm_response =  self.llm.gpt_infer(SUBTASK_DETECTION['system'], SUBTASK_DETECTION['user'].format(instruction))
        # 尝试将 LLM 的 JSON 字符串响应解析为 Python 对象
        subtasks_data = json.loads(llm_response)
        # 将字典列表转换为 Subtask 对象列表
        subtask_queue = [Subtask(**data) for data in subtasks_data]
        return subtask_queue
    
    # --- 新增：管理 subtask_queue 的方法 ---
    def set_subtask_queue(self, subtask_queue: List[Subtask]):
        """接收并存储解析好的子任务队列"""
        self.current_subtask_queue = subtask_queue
        # 可选：重置轨迹树，为新队列做准备
        # self.trajectory_tree_root = None
        # self.current_trajectory_node = None
        print(f"Open_Nav: Subtask queue set with {len(self.current_subtask_queue)} tasks.")

    def get_current_subtask(self) -> Subtask:
        """获取当前子任务 (队列的第一个元素)"""
        if self.current_subtask_queue:
            return self.current_subtask_queue[0]
        else:
            # 如果队列空了，返回一个表示结束的子任务
            return Subtask(action="finished")
        
    def mark_subtask_done(self):
        """当一个子任务完成时，将其从队列中移除"""
        if self.current_subtask_queue:
            completed_task = self.current_subtask_queue.pop(0)
            print(f"Open_Nav: Marked subtask as done: {completed_task}")
            print(f"Open_Nav: Remaining subtasks ({len(self.current_subtask_queue)}):")
            for i, task in enumerate(self.current_subtask_queue):
                print(f"  {i+1}. {task}")
        else:
            print("Open_Nav: Warning: Attempted to mark subtask done, but queue is empty.")
    
    
    # =============================
    # ===== Visual Perception =====
    # =============================







    # TODO：使用图来表征环境

    def observe_environment(self, logger, current_step, images_list):
        '''
        返回: 方法最终返回两个结果：
            observe_results (list): 包含所有方向观察结果的列表，按处理顺序排列。
            observe_dict (dict): 键为方向索引，值为对应观察结果的字典，方便通过索引快速查找。
            之后调用的都是observe_dict，没有再调用observe_results
        '''        
        observe_results = []
        observe_dict = {}
        for direction_idx, direction_image in images_list.items(): 
            # observe_view实现在api.py的spacialClient类中
            observe_result = self.spatial.observe_view(logger, current_step, direction_idx, direction_image)
            logger.info(observe_result)
            observe_results.append(observe_result) 
            observe_dict[direction_idx] = observe_result
        return observe_results, observe_dict
    
    # ===================================
    # ===== Progress Estimation =========
    # ===================================

    # TODO：使用队列来存储并监控进度


        

    # --- 新增：管理 trajectory_tree 构建所需状态的方法 ---
    def set_candidates_for_tree_building(self, observe_dict: Dict[str, str]):
        """临时存储当前步骤的所有候选航点观察，用于后续构建 trajectory tree 节点"""
        self.candidates_from_last_step = observe_dict.copy()
        # print(f"Open_Nav: Candidates for tree building set: {list(self.candidates_from_last_step.keys())}")

    def update_trajectory_tree(self, chosen_viewpoint_id: str, 
                              observation_at_chosen_vp: str, 
                              thought_when_chosen: str,
                              subtask_when_chosen: Subtask):
        """
        根据上一步的决策结果更新导航轨迹树。
        """
        # 1. 创建代表新选择航点的节点
        # 注意：viewpoint_id 在 TrajectoryTreeNode 中被定义为 int，需要转换
        try:
            vp_id_int = int(chosen_viewpoint_id)
        except ValueError:
            print(f"Warning: Could not convert VP ID '{chosen_viewpoint_id}' to int. Using 0.")
            vp_id_int = 0
            
        new_node = TrajectoryTreeNode(
            viewpoint_id=vp_id_int,
            observation = observation_at_chosen_vp,
            thought = thought_when_chosen,
            subtask_at_time = subtask_when_chosen
        )
        
        # 2. 如果是第一步，初始化树
        if self.trajectory_tree_root is None:
            self.trajectory_tree_root = new_node
            self.current_trajectory_node = self.trajectory_tree_root
            print(f"Open_Nav: Initialized trajectory tree with root node: VP {chosen_viewpoint_id}")
        else:
            # 3. 否则，将新节点作为当前节点的子节点
            if self.current_trajectory_node:
                self.current_trajectory_node.add_child(new_node)
                # 更新当前节点为新添加的节点
                self.current_trajectory_node = new_node
                print(f"Open_Nav: Added node VP {chosen_viewpoint_id} as child of VP {self.current_trajectory_node.parent.viewpoint_id if self.current_trajectory_node.parent else 'None'}")
            else:
                 print("Open_Nav: Error: current_trajectory_node is None when trying to add child.")

        # 4. 清理临时存储
        self.candidates_from_last_step = {}
        # print("Open_Nav: Cleared candidates for tree building.")


    
    def initialize_scene_graph(self, start_waypoint_id: str, direction_image: dict, logger):
        """
        (占位符/简化版) 在 Episode 开始时，基于初始观察初始化 Scene Graph。
        当前实现：仅添加起始航点节点。
        未来可以集成更复杂的初始化逻辑。

        Args:
            start_waypoint_id (str): 起始航点的 ID。
            direction_image (dict): 起始航点的图像数据。
            logger: 日志记录器。
        """
        log_prefix = "[initialize_scene_graph]"
        logger.info(f"{log_prefix} Initializing SceneGraph with start waypoint {start_waypoint_id}")
        self.scene_graph = SceneGraph()
        
        # 创建代表起始航点的节点
        start_wp_node_id = f"wp_{start_waypoint_id}"
        start_wp_node = SceneNode(
            id=start_wp_node_id,
            type=NodeType.WAYPOINT,
            attributes={'viewpoint_id': int(start_waypoint_id) if start_waypoint_id.isdigit() else start_waypoint_id}
        )
        self.scene_graph.add_node(start_wp_node)
        logger.info(f"{log_prefix} Added start waypoint node: {start_wp_node}")


    def update_scene_graph(self, current_waypoint_id: str, direction_image: dict, logger):
        """
        在智能体移动到新航点后，更新内部的 SceneGraph。

        Args:
            current_waypoint_id (str): 智能体当前所在的航点 ID。
            direction_image (dict): 包含该航点 'rgb' 和 'depth' 图像的字典。
            logger: 日志记录器。
        """
        log_prefix = "[update_scene_graph]"
        if not self.is_scene_graph_initialized():
            logger.warning(f"{log_prefix} SceneGraph is not initialized. Initializing with current waypoint.")
            # 如果尚未初始化（例如，在第一帧），则进行初始化
            self.initialize_scene_graph(current_waypoint_id, direction_image, logger)
            # 注意：initialize_scene_graph 只添加了航点节点，我们仍需要更新该航点的观察信息。
            # 因此，代码会继续执行下去，调用 spatialClient 来获取观察并更新。

        logger.info(f"{log_prefix} Updating SceneGraph for waypoint {current_waypoint_id}.")

        # 1. (可选) 获取当前航点的局部子图，用于与新观察进行比对
        #    这有助于实现更高级的融合逻辑（如更新置信度、确认存在等）
        #    对于基础实现，我们可以暂时不传入子图。
        current_subgraph: Optional[SceneGraph] = None
        # 如果需要子图，可以这样获取：
        # try:
        #     current_subgraph = self.scene_graph.get_subgraph_around_node(f"wp_{current_waypoint_id}", radius=2)
        # except Exception as e:
        #     logger.warning(f"{log_prefix} Could not get subgraph: {e}. Proceeding without it.")

        # 2. 调用 spatialClient 的新方法获取更新
        try:
            new_nodes, new_edges = self.spatial.update_scene_graph_from_observation(
                logger=logger,
                current_waypoint_id=current_waypoint_id,
                direction_image=direction_image,
                current_subgraph=current_subgraph # 可以传入 None 或实际的子图
            )
        except Exception as e:
            logger.error(f"{log_prefix} Error calling spatialClient.update_scene_graph_from_observation: {e}")
            return # 如果感知失败，则不进行更新

        # 3. 将返回的节点和边整合到主 SceneGraph 中
        nodes_added = 0
        edges_added = 0
        nodes_updated = 0 # 用于记录更新的节点数
        edges_updated = 0 # 用于记录更新的边数

        for node in new_nodes:
            # 检查节点是否已存在
            if not self.scene_graph.graph.has_node(node.id):
                self.scene_graph.add_node(node)
                nodes_added += 1
                logger.debug(f"{log_prefix} Added new node to SceneGraph: {node}")
            else:
                # 节点已存在，更新其属性（例如，增加观察次数）
                # 注意：NetworkX 节点属性更新需要直接操作 graph.nodes[node_id]['data']
                existing_node_data = self.scene_graph.graph.nodes[node.id].get('data')
                if existing_node_data:
                    # 简单示例：增加观察计数
                    existing_node_data.attributes['observation_count'] = existing_node_data.attributes.get('observation_count', 0) + 1
                    # 可以在这里添加更多属性更新逻辑
                    nodes_updated += 1
                    logger.debug(f"{log_prefix} Updated existing node in SceneGraph: {node.id}")

        for edge in new_edges:
            # 检查边是否已存在
            if not self.scene_graph.graph.has_edge(edge.source_id, edge.target_id):
                self.scene_graph.add_edge(edge)
                edges_added += 1
                logger.debug(f"{log_prefix} Added new edge to SceneGraph: {edge}")
            else:
                # 边已存在，更新其属性（例如，提高置信度）
                # 注意：NetworkX 边属性更新可以直接通过字典赋值
                existing_edge_data = self.scene_graph.graph[edge.source_id][edge.target_id]
                # 简单示例：提高置信度（但不超过1）
                old_conf = existing_edge_data.get('confidence', 0.5)
                new_conf = min(1.0, old_conf + 0.1) # 增加0.1置信度
                existing_edge_data['confidence'] = new_conf
                # 可以在这里添加更多属性更新逻辑
                edges_updated += 1
                logger.debug(f"{log_prefix} Updated existing edge in SceneGraph: {edge.source_id} -> {edge.target_id}. New confidence: {new_conf}")

        logger.info(f"{log_prefix} SceneGraph update complete. "
                    f"Added {nodes_added} nodes, Updated {nodes_updated} nodes, "
                    f"Added {edges_added} edges, Updated {edges_updated} edges.")

    def is_scene_graph_initialized(self) -> bool:
        """检查 Scene Graph 是否已初始化。"""
        return self.scene_graph is not None and self.scene_graph.graph.number_of_nodes() > 0




    def is_subtask_completed(self, subtask: Subtask, observation: str, logger) -> bool:
        """
        (核心逻辑) 利用 LLM 判断一个子任务是否基于观察完成。
        ...
        """
        log_prefix = "[is_subtask_completed]"
        logger.info(f"{log_prefix} Checking completion for subtask: {subtask}")
        logger.info(f"{log_prefix} Against observation: {observation[:100]}...")

        # 1. 准备 Prompt
        # 将 Subtask 对象转换为易于 LLM 理解的字符串
        subtask_str = (
            f"Action: '{subtask.action or 'N/A'}', "
            f"Direction: '{subtask.direction or 'N/A'}', "
            f"Preposition: '{subtask.preposition or 'N/A'}', "
            f"Landmark: '{subtask.landmark or 'N/A'}'"
        )

        # 2. 调用 LLM
        try:
            logger.info(f"{log_prefix} Prompting LLM for subtask completion judgment...")
            # llm_response = self.llm.gpt_infer(system_prompt, user_prompt)
            # 或者更简洁地直接使用导入的字典
            llm_response = self.llm.gpt_infer(
                SUBTASK_COMPLETION_JUDGE['system'],
                SUBTASK_COMPLETION_JUDGE['user'].format(subtask_str=subtask_str, observation=observation)
            )
            
            # 3. 解析 LLM 的响应 (后续代码保持不变)
            logger.info(f"{log_prefix} LLM Response: {llm_response}")
            response_clean = llm_response.strip().lower()
            if "true" in response_clean:
                is_completed = True
            elif "false" in response_clean:
                is_completed = False
            else:
                logger.warning(f"{log_prefix} LLM response was unclear ('{llm_response}'). Defaulting to NOT completed.")
                is_completed = False
                
        except Exception as e:
            logger.error(f"{log_prefix} Error calling LLM for completion check: {e}")
            logger.info(f"{log_prefix} Defaulting to NOT completed due to error.")
            is_completed = False

        logger.info(f"{log_prefix} Judged subtask completion as: {is_completed}")
        return is_completed


    # =================================
    # ===== Move to next position =====
    # =================================
    # TODO: 修改方法签名以使用 current_subtask

    def move_to_next_vp(self, logger, current_step, current_subtask: Subtask, observation, observe_dict):    
        """
        基于当前子任务、当前环境的 Scene Graph 子图，预测下一步应该去哪个航点。

        Args:
            logger: 日志记录器。
            current_step (int): 当前导航步数。
            current_subtask (Subtask): 当前需要完成的子任务。
            observation (str/list): 所有航点的观察结果（可能未使用）。
            observe_dict (dict): 航点索引到观察结果的字典（可能未使用或仅用于获取候选列表）。

        Returns:
            tuple: (effective_prediction, thought_list, break_flag)
        """
        log_prefix = "[move_to_next_vp]"
        logger.info(f"{log_prefix} Starting next viewpoint prediction (Step {current_step}).")

        break_flag = True

        # 1. 获取候选航点ID列表 (这一步不变)
        candidate_vp_ids = list(observe_dict.keys())
        
        # 2. 将当前子任务转换为易于Prompt使用的字符串格式 (这一步不变)
        current_subtask_str = (
            f"Action: {current_subtask.action or 'N/A'}, "
            f"Direction: {current_subtask.direction or 'N/A'}, "
            f"Preposition: {current_subtask.preposition or 'N/A'}, "
            f"Landmark: {current_subtask.landmark or 'N/A'}"
        )
        
        # --- 新增：获取 Scene Graph 信息 ---
        scene_graph_context = "No scene graph context available."
        # 假设我们知道当前智能体所在的航点 ID
        # 这可以从 self.current_trajectory_node 获取，或者作为参数传入
        # 注意 ID 格式的统一 (例如，trajectory node 存的是整数 0, SceneGraph node ID 是字符串 "wp_0")
        current_agent_wp_id_int = None
        if self.current_trajectory_node:
            current_agent_wp_id_int = self.current_trajectory_node.viewpoint_id
        
        if self.scene_graph and current_agent_wp_id_int is not None:
            try:
                current_wp_node_id_sg = f"wp_{current_agent_wp_id_int}"
                
                # 3. 获取以当前智能体所在航点为中心的局部子图
                #    radius 可以根据需要调整
                current_subgraph = self.scene_graph.get_subgraph_around_node(current_wp_node_id_sg, radius=2) 
                
                # 4. 将子图序列化为文本
                scene_graph_context = current_subgraph.serialize_to_text(detail_level="concise") 
                logger.debug(f"{log_prefix} Retrieved Scene Graph context for current WP {current_wp_node_id_sg}: {scene_graph_context[:200]}...")
            except Exception as e:
                logger.warning(f"{log_prefix} Could not retrieve Scene Graph context for current WP: {e}. Proceeding without it.")
                scene_graph_context = "Error retrieving scene graph context."
        else:
            logger.info(f"{log_prefix} Scene Graph or current agent waypoint not available. Proceeding without Scene Graph context.")
        # --- 新增结束 ---

        # 5. 调用 LLM 进行推理
        #    将候选航点、当前步数、当前子任务、以及最重要的 Scene Graph 上下文传递给 LLM
        for i in range(2): # retry twice
            effective_prediction, thought_list = [], []
            batch_responses = self.llm.gpt_infer(
                NAVIGATOR['system'], 
                NAVIGATOR['user'].format(
                    candidate_vp_ids,       # 候选航点列表
                    current_step,           # 当前步数
                    current_subtask_str,    # 结构化的当前子任务
                    scene_graph_context     # 新增：结构化的 Scene Graph 上下文 (来自当前航点)
                    # observation           # 原始观察 (可以选择性保留或移除)
                ),
                num_output=3
            )
            # 6. 解析 LLM 的响应 (这部分逻辑保持不变)
            for decision_reasoning in batch_responses:
                logger.info(decision_reasoning)
                if "Prediction:" not in decision_reasoning:
                    continue
                logger.info(f"{log_prefix}================retry id {i} in pred_vp==========")
                pred_thought = decision_reasoning.split("Prediction:")[0].strip()
                pred_text = decision_reasoning.split("Prediction:")[1].strip()
                pred_vp = re.sub(r'[\"\'\n\.\*\`\`\`]', '', pred_text).strip()
                effective_prediction.append(pred_vp)
                thought_list.append(pred_thought)
        return effective_prediction, thought_list, break_flag

    
    # =========================
    # ===== Test Decision =====
    # =========================
    def thought_fusion(self, logger, predictions, thoughts):
        matched_dict = dict()
        for pred, thought in zip(predictions, thoughts):
            if pred not in matched_dict.keys():
                matched_dict[pred] = []
            matched_dict[pred].append(thought)
            
        for key, value in matched_dict.items():
            multiple_thoughts = "; ".join(["Thought "+str(idx+1)+": "+thought for idx, thought in enumerate(value)])
            one_thought = self.llm.gpt_infer(THOUGHT_FUSION['system'], THOUGHT_FUSION['user'].format(multiple_thoughts))
            logger.info(f"Pred viewpoint ID: {key} Fused Thought: {one_thought}")
            matched_dict[key] = one_thought 
        return matched_dict 
    
    # TODO: 修改方法签名以使用 current_subtask
    def test_decisions(self, logger, fused_pred_thought, observation, current_subtask: Subtask, error_number, observe_dict):
        """
        从融合后的候选决策中，做出最终的选择。

        Args:
            logger: 日志记录器。
            fused_pred_thought (dict): 航点到融合思考的字典。
            observation: 当前整体观察。
            current_subtask (Subtask): 当前子任务。
            error_number (int): 错误计数。
            observe_dict (dict): 候选航点观察字典。

        Returns:
            tuple: (next_vp, thought, error_number)
        """
        try:
            # 清理无效的航点索引
            for fused_key in list(fused_pred_thought.keys()):
                if len(fused_key) > 2:
                    fused_pred_thought.pop(fused_key)
                    
            if not fused_pred_thought:
                raise ValueError("Error in fused_thought key")
                
            if len(fused_pred_thought.keys()) == 1:
                for key, value in fused_pred_thought.items():
                    return key, value, error_number
            else:
                # 如果有多个候选，则再次调用LLM进行最终决策
                fused_pred_thought_ = "; ".join(["Direction Viewpoint ID: "+key+" Thought: "+value for key, value in fused_pred_thought.items()])

                # TODO：将当前子任务信息加入Prompt
                current_subtask_str = (
                    f"Action: {current_subtask.action or 'N/A'}, "
                    f"Landmark: {current_subtask.landmark or 'N/A'}"
                )

                for i in range(2): 
                    logger.info(f"========== {i} retry in test decision==========")
                    next_vp = self.llm.gpt_infer(DECISION_TEST['system'], DECISION_TEST['user'].format(fused_pred_thought.keys(), observation, current_subtask_str, fused_pred_thought_))
                    logger.info(f"Next predicted action is {next_vp}")
                    if re.search(r'\D', next_vp):
                        next_vp = re.search(r'\d+', next_vp).group() 
        
            logger.info(f"In test decision the predicted direction: {next_vp}")
            logger.info(f"In test decision the predicted thought: {fused_pred_thought[next_vp]}")
            return next_vp, fused_pred_thought[next_vp], error_number
        except Exception as e:
            logger.info(f"Error in test decision {e}")
            error_number += 1
            logger.info(f"Error number is {error_number}")
            
            if error_number >= 2: 
                error_number = 0 
                if fused_pred_thought and all(len(key) < 2 for key in fused_pred_thought):
                    logger.info(f"Random choice a next predicted action {next_vp} in fused_pred_thought, error number reset to {error_number}")
                    next_vp, _ = random.choice(list(fused_pred_thought.items()))
                    return next_vp, fused_pred_thought[next_vp], error_number
                else:
                    next_vp, observe_description = random.choice(list(observe_dict.items()))
                    logger.info(f"Random choice a next predicted action {next_vp}, error number reset to {error_number}")
                    return next_vp, observe_description, error_number
            return "error_next_vp", "None", error_number


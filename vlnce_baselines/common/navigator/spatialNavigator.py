'''vlnce_baselines/common/navigator/spatialNavigator.py'''
import re   # 正则表达式
import random
import json # get_subtasks中解析用
from vlnce_baselines.common.navigator.api import *
from vlnce_baselines.common.navigator.prompts import *

from typing import List, Dict, Optional # 添加类型注解所需

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
        基于当前子任务和环境观察，预测下一步应该去哪个航点。

        Args:
            logger: 日志记录器。
            current_step (int): 当前导航步数。
            current_subtask (Subtask): 当前需要完成的子任务。
            observation (list): 所有航点的观察结果列表（可能未使用，取决于Prompt设计）。
            observe_dict (dict): 航点索引到观察结果的字典。

        Returns:
            tuple: (effective_prediction, thought_list, break_flag)
        """
        break_flag = True   # TODO：注意一下break_flag

        # 提取候选航点ID列表
        candidate_vp_ids = list(observe_dict.keys())
        
        # 将当前子任务转换为易于Prompt使用的字符串格式
        # 你可以根据需要调整这个格式
        current_subtask_str = (
            f"Action: {current_subtask.action or 'N/A'}, "
            f"Direction: {current_subtask.direction or 'N/A'}, "
            f"Preposition: {current_subtask.preposition or 'N/A'}, "
            f"Landmark: {current_subtask.landmark or 'N/A'}"
        )

        for i in range(2): # retry twice
            effective_prediction, thought_list = [], []
            batch_responses = self.llm.gpt_infer(
                NAVIGATOR['system'], 
                NAVIGATOR['user'].format(
                    candidate_vp_ids,       # 候选航点
                    current_step,           # 当前步数
                    current_subtask_str,    # 结构化的当前子任务
                    observation             # 观察
                ),
                num_output=3
            )
            for decision_reasoning in batch_responses:
                logger.info(decision_reasoning)
                if "Prediction:" not in decision_reasoning:
                    continue
                logger.info(f"================retry id {i} in pred_vp==========")
                logger.info(decision_reasoning)
                pred_thought = decision_reasoning.split("Prediction:")[0].strip()
                # 提取预测的航点ID，增强鲁棒性
                pred_text = decision_reasoning.split("Prediction:")[1].strip()
                # 移除常见的标点符号
                pred_vp = re.sub(r'[\"\'\n\.\*\`\`\`]', '', pred_text).strip()
                effective_prediction.append(pred_vp)                            # TODO：⤒ 这里引号对吗
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


import json
import sys
import jsonlines
import os
import time
import warnings
from collections import defaultdict
from typing import Dict, List
from PIL import Image
import requests
from openai import OpenAI

# for navigator      
from vlnce_baselines.common.navigator.spatialNavigator import *
import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as distr
import torch.multiprocessing as mp
import gzip
import math
from copy import deepcopy

import tqdm
from gym import Space
from habitat import Config, logger
from habitat.utils.visualizations.utils import append_text_to_image
from habitat_baselines.common.base_il_trainer import BaseILTrainer
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.environments import get_env_class
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_batch,
    apply_obs_transforms_obs_space,
    get_active_obs_transforms,
)
from habitat_extensions.measures import Position
from habitat_baselines.common.tensorboard_utils import TensorboardWriter
from habitat_baselines.utils.common import batch_obs, generate_video
from habitat_baselines.utils.common import (
    get_checkpoint_id,
    poll_checkpoint_folder,
)

from habitat_extensions.utils import observations_to_image
from vlnce_baselines.common.aux_losses import AuxLosses
from vlnce_baselines.common.env_utils import (
    construct_envs_auto_reset_false,
    construct_envs,
    is_slurm_batch_job,
)
from vlnce_baselines.common.utils import *

from habitat_extensions.measures import NDTW
from fastdtw import fastdtw

from ..utils import get_camera_orientations
from ..models.utils import (
    length2mask, dir_angle_feature, dir_angle_feature_with_ele,
)

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import tensorflow as tf  # noqa: F401

class BaseVLNCETrainerLLM(BaseILTrainer):
    r"""A base trainer for VLN-CE imitation learning."""
    supported_tasks: List[str] = ["VLN-v0"]

    def __init__(self, config=None):
        super().__init__(config)
        self.policy = None
        self.device = (
            torch.device("cuda", self.config.TORCH_GPU_ID)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        self.obs_transforms = []
        self.start_epoch = 0
        self.step_id = 0

    def _initialize_policy(
        self,
        config: Config,
        load_from_ckpt: bool,
        observation_space: Space,
        action_space: Space,
    ) -> None:
        policy = baseline_registry.get_policy(self.config.MODEL.policy_name)
        self.policy = policy.from_config(
            config=config,
            observation_space=observation_space,
            action_space=action_space,
        )
        ''' initialize the waypoint predictor here '''
        from waypoint_prediction.TRM_net import BinaryDistPredictor_TRM
        self.waypoint_predictor = BinaryDistPredictor_TRM(device=self.device)
        self.waypoint_predictor.load_state_dict(
            torch.load(
                './waypoint_prediction/checkpoints/check_val_best_avg_wayscore',
                map_location = torch.device('cpu'),
            )['predictor']['state_dict']
        )
        for param in self.waypoint_predictor.parameters():
            param.requires_grad = False

  
        self.policy.to(self.device)
        self.waypoint_predictor.to(self.device)
        self.num_recurrent_layers = self.policy.net.num_recurrent_layers

        logger.info("Finished setting up waypoint_predictor.")

    def load_checkpoint(self, checkpoint_path, *args, **kwargs) -> Dict:
        return torch.load(checkpoint_path, *args, **kwargs)

    @staticmethod
    def _pause_envs(
        envs_to_pause,
        envs,
        not_done_masks,
        prev_actions,
        batch,
        rgb_frames=None,
    ):
        if len(envs_to_pause) > 0:
            state_index = list(range(envs.num_envs))
            for idx in reversed(envs_to_pause):
                state_index.pop(idx)
                envs.pause_at(idx)
                
            not_done_masks = not_done_masks[state_index]
            prev_actions = prev_actions[state_index]

            for k, v in batch.items():
                batch[k] = v[state_index]

            if rgb_frames is not None:
                rgb_frames = [rgb_frames[i] for i in state_index]

        return (
            envs,
            not_done_masks,
            prev_actions,
            batch,
            rgb_frames,
        )
        
    def generate_input(self, observations):
        instruction = observations['instruction']['text']
        image_dict = {} 
        rgb_image_dict = {}
        depth_image_dict = {}
        rgb_index = 0
        depth_index = 0
        for key in observations.keys():
            image_path = "./image_show/"
            if 'rgb' in key:
                image_path += f"{key}.jpg"
                image = Image.fromarray(observations[key], mode="RGB")
                dir_name = os.path.dirname(image_path)
                if not os.path.exists(dir_name):
                    os.makedirs(dir_name)
                image.save(image_path, format="JPEG")
                rgb_image_dict[str(rgb_index)] = Image.open(image_path)
                rgb_index += 1
            if 'depth' in key:
                image_path += f"{key}.jpg"
                if observations[key].ndim == 3 and observations[key].shape[-1] == 1:
                    depth_map = observations[key].squeeze(-1)
                depth_img = (255 * (depth_map - np.min(depth_map)) / (np.max(depth_map) - np.min(depth_map))).astype(np.uint8)
                image = Image.fromarray(depth_img)
                dir_name = os.path.dirname(image_path)
                if not os.path.exists(dir_name):
                    os.makedirs(dir_name)
                image.save(image_path)
                depth_image_dict[str(depth_index)] = Image.open(image_path)
                depth_index += 1
        for index in rgb_image_dict:
            image_dict[index] = {
                'rgb': rgb_image_dict[index],
                'depth': depth_image_dict[index]
            }
            
        return instruction, image_dict
    
    def construct_image_dicts(self, batch_distance, batch_angles, image_dict):
        waypoint_distances = {}
        waypoint_radius = {}
        waypoint_images = {}
        angles = batch_angles[-1]
        for angle_idx in range(len(angles)):
            angle = angles[angle_idx]
            angle_deg = np.rad2deg(angle)
            if 0 < angle_deg <= 30:
                waypoint_images['1'] = image_dict['1']
                waypoint_distances['1'] = batch_distance[angle_idx]
                waypoint_radius['1'] = angles[angle_idx]
            elif 30 < angle_deg <= 60:
                waypoint_images['2'] = image_dict['2']
                waypoint_distances['2'] = batch_distance[angle_idx]
                waypoint_radius['2'] = angles[angle_idx]
            elif 60 < angle_deg <= 90:
                waypoint_images['3'] = image_dict['3']
                waypoint_distances['3'] = batch_distance[angle_idx]
                waypoint_radius['3'] = angles[angle_idx]
            elif 90 < angle_deg <= 120:
                waypoint_images['4'] = image_dict['4']
                waypoint_distances['4'] = batch_distance[angle_idx]
                waypoint_radius['4'] = angles[angle_idx]
            elif 120 < angle_deg <= 150:
                waypoint_images['5'] = image_dict['5']
                waypoint_distances['5'] = batch_distance[angle_idx]
                waypoint_radius['5'] = angles[angle_idx]
            elif 150 < angle_deg <= 180:
                waypoint_images['6'] = image_dict['6']
                waypoint_distances['6'] = batch_distance[angle_idx]
                waypoint_radius['6'] = angles[angle_idx]
            elif 180 < angle_deg <= 210:
                waypoint_images['7'] = image_dict['7']
                waypoint_distances['7'] = batch_distance[angle_idx]
                waypoint_radius['7'] = angles[angle_idx]
            elif 210 < angle_deg <= 240:
                waypoint_images['8'] = image_dict['8']
                waypoint_distances['8'] = batch_distance[angle_idx]
                waypoint_radius['8'] = angles[angle_idx]
            elif 240 < angle_deg <= 270:
                waypoint_images['9'] = image_dict['9']
                waypoint_distances['9'] = batch_distance[angle_idx]
                waypoint_radius['9'] = angles[angle_idx]
            elif 270 < angle_deg <= 300:
                waypoint_images['10'] = image_dict['10']
                waypoint_distances['10'] = batch_distance[angle_idx]
                waypoint_radius['10'] = angles[angle_idx]
            elif 300 < angle_deg <= 330:
                waypoint_images['11'] = image_dict['11']
                waypoint_distances['11'] = batch_distance[angle_idx]
                waypoint_radius['11'] = angles[angle_idx]
            else:
                waypoint_images['0'] = image_dict['0']  
                waypoint_distances['0'] = batch_distance[angle_idx]
                waypoint_radius['0'] = angles[angle_idx]
                
        return waypoint_images, waypoint_radius, waypoint_distances
    

    def _eval_llm(
        self,
    ) -> None:
        r"""Evaluation.

        Args:
            writer: tensorboard writer object
            checkpoint_index: index of the current checkpoint

        Returns:
            None
        """
        # 1. 克隆配置以避免修改原始配置
        config = self.config.clone()

        # 2. 解冻配置以便修改
        config.defrost()
        # 禁用数据集打乱，确保评估的一致性
        config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = False
        # 允许场景重复，不限制重复步数
        config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS = (
            -1
        )   # TODO：括号换行是多余的
        # 3. 如果需要生成视频，添加额外的测量项（如地图、碰撞）
        if len(config.VIDEO_OPTION) > 0:
            config.defrost()    # TODO：这行是多余的，因为上面已经解冻了
            config.TASK_CONFIG.TASK.MEASUREMENTS.append("TOP_DOWN_MAP_VLNCE")
            config.TASK_CONFIG.TASK.MEASUREMENTS.append("COLLISIONS")
        # 4. 重新冻结配置
        config.freeze()
        
        # 5. 检查是否已存在评估结果文件，避免重复计算
        if config.EVAL.SAVE_RESULTS:
            fname = os.path.join(
                config.RESULTS_DIR,
                f"stats_ckpt_{config.TASK_CONFIG.DATASET.SPLIT}.json",
            )
            if os.path.exists(fname):
                print(f"skipping -- evaluation exists. File path: {fname}")
                # 询问用户是否覆盖
                user_input = input("Do you want to overwrite the results? (yes/no): ").strip().lower()
                if user_input != "yes":
                    print("Skipping evaluation.")
                    return
                else:
                    print("Overwriting previous results...")
                
        # 6. 构建评估环境
        #    - 使用配置和环境类创建环境
        #    - auto_reset_done=False: 环境不会在episode结束时自动重置，由代码手动控制
        #    - episodes_allowed=self.traj: 只评估指定的episode（通过collect_val_traj获取）
        envs = construct_envs(
            config, get_env_class(config.ENV_NAME),
            auto_reset_done=False,
            episodes_allowed=self.traj
        ) 

        #envs.number_of_episodes = [1] # set the number of episodes

        # 7. 计算数据集总长度（所有环境中的episode数之和）
        dataset_length = sum(envs.number_of_episodes) 
        print('local rank:', self.local_rank, '|', 'dataset length:', dataset_length)

        # 8. 获取并应用观测变换（如图像大小调整）
        obs_transforms = get_active_obs_transforms(config) 
        observation_space = apply_obs_transforms_obs_space(
            envs.observation_spaces[0], obs_transforms
        )

        # 9. 初始化策略网络和航点预测器
        #    - load_from_ckpt=False: 不从检查点加载（因为是零样本推理）
        #    - observation_space, action_space: 用于构建网络
        self._initialize_policy(
            config,
            load_from_ckpt=False,
            observation_space=observation_space,
            action_space=envs.action_spaces[0],
        )

        # 10. 将模型设置为评估模式（关闭dropout, batchnorm更新等）
        self.policy.eval() 
        self.waypoint_predictor.eval()

        # 11. 重置环境，获取初始观测
        observations = envs.reset()
        
        # 12. 生成输入图像（可能用于调试或外部模型）并提取指令token
        instruction, images_list = self.generate_input(observations[-1])
        observations = extract_instruction_tokens(
            observations, self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID
        ) 
        # 13. 将观测打包成批次，并应用观测变换
        batch = batch_obs(observations, self.device) 
        batch = apply_obs_transforms_batch(batch, obs_transforms) 

        # 14. 初始化未完成掩码（用于并行环境管理）
        not_done_masks = torch.zeros(
            envs.num_envs, 1, dtype=torch.uint8, device=self.device
        ) 

        # 15. 初始化用于存储统计信息的字典和视频帧缓冲区
        stats_episodes = {} # 存储每个episode的评估指标
        rgb_frames = [[] for _ in range(envs.num_envs)] # 存储每个环境的视频帧
        if len(config.VIDEO_OPTION) > 0:
            os.makedirs(config.VIDEO_DIR, exist_ok=True)    # 创建视频目录

        # 16. 确定要评估的episode数量
        if config.EVAL.EPISODE_COUNT == -1:
            episodes_to_eval = sum(envs.number_of_episodes)  # 评估所有
        else:
            episodes_to_eval = min(
                config.EVAL.EPISODE_COUNT, sum(envs.number_of_episodes)  # 评估指定数量
            )

        # 17. 初始化进度条和日志字符串
        pbar = tqdm.tqdm(total=episodes_to_eval) if config.use_pbar else None
        log_str = (
            " [Episodes evaluated: {evaluated}/{total}]"
            " [Time elapsed (s): {time}]"
        )
        start_time = time.time()

        # 18. 设置日志记录器
        # set up the logger
        log_file = "./navigator_log.log"
        if os.path.exists(log_file): os.remove(log_file)    # 清除旧日志
        import logging
        logging.basicConfig(
            format='%(asctime)s - %(filename)s/%(funcName)s[line:%(lineno)d] - %(levelname)s: %(message)s',
            datefmt="%Y-%m-%d %H:%M:%S",
            level=os.environ.get("LOGLEVEL", "INFO").upper(),
            stream=sys.stdout,
            filemode="a"
        )
        nav_logger = logging.getLogger("vln_logger")     # 创建专门的导航日志记录器
        nav_logger.addHandler(logging.FileHandler(filename=log_file))   # 添加文件处理器
        
        # 19. 设置动作缓存（避免对相同指令重复调用LLM）
        dataset_name = "R2R"
        if not os.path.exists(f"cache_files/{dataset_name}"):
            os.makedirs(f"cache_files/{dataset_name}")

        subtasks_cache_path = f"./cache_files/{dataset_name}/subtasks_cache.json"
        if os.path.exists(subtasks_cache_path): 
            with open(subtasks_cache_path, "r", encoding="utf-8") as file:
                subtasks_cache = json.load(file)
        else:
            subtasks_cache = {} 
        
        # 20. 初始化核心导航器（Open_Nav类实例）
        #     这是执行Spatial-Temporal CoT推理的地方
        navigator = Open_Nav(self.device,config.LLM, config.API_KEY)
        current_step = 0
        nav_history = []
        error_number = 0

        # 21. 主评估循环：只要还有环境在运行且未达到评估数量上限
        while envs.num_envs > 0 and len(stats_episodes) < episodes_to_eval:
            current_episodes = envs.current_episodes()
            positions = []; headings = []
            for ob_i in range(len(current_episodes)): 
                agent_state_i = envs.call_at(ob_i,
                        "get_agent_info", {})
                positions.append(agent_state_i['position'])
                headings.append(agent_state_i['heading'])

            # ==========Navigator start==========
            # 21.2 记录当前episode ID和指令到日志
            nav_logger.info(f"==================== The current episode id is {current_episodes[0].episode_id} ====================")
            nav_logger.info("Instruction: "+instruction)

            # 21.3 从缓存获取或通过LLM生成动作序列和地标
            subtasks = []
            if instruction not in subtasks_cache.keys():
                # 调用 Open_Nav 的方法，内部会调用LLM进行推理
                subtasks = navigator.get_subtasks(instruction)

                # 缓存结果
                # --- 序列化 Subtask 对象以便缓存 ---
                # 将 Subtask 对象列表转换为字典列表
                subtasks_for_cache = [task.to_dict() for task in subtasks]
                subtasks_cache[instruction] = {"subtasks": subtasks_for_cache}
                with open(subtasks_cache_path, "w", encoding="utf-8") as f2:
                    json.dump(subtasks_cache, f2, indent=2)
            else:
                # --- 反序列化缓存的字典列表为 Subtask 对象列表 ---
                # 从缓存中加载字典列表
                subtasks_from_cache = subtasks_cache[instruction]["subtasks"]
                # 将字典列表转换回 Subtask 对象列表
                subtasks = [Subtask(**data) for data in subtasks_from_cache]
            nav_logger.info(f"Parsed Subtask Queue (Length: {len(subtasks)}):")
            for i, task in enumerate(subtasks):
                 nav_logger.info(f"  {i+1}. {task}") # 这里会调用 Subtask 的 __str__ 方法

            
            # 21.4 根据动作序列长度确定导航步数上限
            step_length = 6 if len(subtasks) <= 6 else 8    # TODO：这是Open-Nav原来的设计，莫名其妙

            stop_flag = False   # 停止标志（代码中似乎未被设置为True）
            current_step += 1
            nav_logger.info(f"-------------------- Step {current_step} --------------------")

            # 21.5 预测候选航点
            with torch.no_grad():   # 禁用梯度计算，因为是推理阶段
                # candidate waypoints prediction
                cand_rgb, cand_depth, \
                cand_direction, cand_mask, candidate_lengths, \
                batch_angles, batch_distances = self.policy.net( 
                    mode = "waypoint",
                    waypoint_predictor = self.waypoint_predictor,
                    observations = batch,
                    in_train = False,
                )
            
            # 21.6 构建航点图像字典（供后续LLM推理使用）
            images_dict, radius_dict, distance_dict = self.construct_image_dicts(batch_distances[-1], batch_angles, images_list)

            # 21.7 调用 Open_Nav 进行环境观察（可能涉及外部模型如RAM/SpatialBot）
            nav_logger.info("========== Get Observation ==========")
            observation, observe_dict = navigator.observe_environment(nav_logger, current_step, images_dict)
            
            
            # --- 获取当前的子任务目标 (替代旧的 history_traj, actions, landmarks, estimation) ---
            nav_logger.info("========== Get Current Subtask ==========")
            current_subtask = navigator.get_current_subtask() # <-- 获取subtasks队列头部的子任务
            nav_logger.info(f"Current Subtask Goal: {current_subtask}")

            # 21.9 如果未停止，执行完整的Spatial-Temporal CoT推理链
            if not stop_flag:
                
                # 21.9.2 预测下一步动作
                nav_logger.info("========== Next Action Prediction ==========")
                predictions, thoughts, break_flag = navigator.move_to_next_vp(nav_logger, current_step, current_subtask, observation, observe_dict)
                                # FIXME：这个break_flag并没有使用？

                # 21.9.3 融合预测和思考
                nav_logger.info("========== Thought ==========")
                fused_pred_thought = navigator.thought_fusion(nav_logger, predictions, thoughts)
                
                # 21.9.4 最终决策（选择一个航点）
                nav_logger.info("========== Test Decision ==========")
                next_vp, thought, error_number = navigator.test_decisions(nav_logger, fused_pred_thought, observation, current_subtask, error_number, observe_dict)
            
            # 22. 尝试执行动作并处理结果           
            try:
                if not stop_flag:
                    # 22.1 构建环境动作（移动到指定航点）
                    env_actions = []
                    env_actions.append({'action':
                        {'action': 4,   # TODO：4？
                        'action_args':{
                            'angle': radius_dict[next_vp],  # 航点的角度
                            'distance': distance_dict[next_vp], # 航点的距离
                        }}})
                    nav_logger.info(f"The final env action: {env_actions}")

                    # 22.2 在环境中执行动作
                    outputs = envs.step(env_actions)
                    
                    # 22.3 更新subtasks队列，
                    curr_observe = observe_dict[next_vp]
                    nav_logger.info("========== Progress Estimation ==========")
                    

                    nav_logger.info("========== save history ==========")
                    nav_history = navigator.save_history(nav_logger, current_step, next_vp, thought, curr_observe, nav_history)
                
                    # 22.4 获取新观测，并重置错误计数
                    observations, _, dones, infos = [list(x) for x in zip(*outputs)]
                    instruction, images_list = self.generate_input(observations[-1])
                    error_number = 0 

                    # 22.5 检查是否达到步数上限，若是则标记为完成，并结束导航
                    # finish navigation
                    if current_step == step_length:
                        dones[0] = True 
                    else:
                        # 22.6 否则更新环境内部路径信息（用于评估指标计算）
                        for j, ob in enumerate(observations):
                            envs.call_at(j, 
                                'change_current_path',
                                {'new_path': ob.pop('positions'),
                                'collisions': ob.pop('collisions')}
                            )
                else:
                    # 如果stop_flag为True，则直接标记为完成（但代码中stop_flag似乎始终为False）
                    dones[0] = True
                
                # 22.7 更新未完成掩码
                not_done_masks = torch.tensor(
                    [[0] if done else [1] for done in dones],
                    dtype=torch.uint8, device=self.device)
                
                # 23. 遍历每个环境，处理已完成的episode
                for i in range(envs.num_envs):
                    
                    if not dones[i]:
                        continue
                    
                    # 23.1 重置episode相关变量
                    current_step = 0
                    nav_history = []

                    # 23.2 计算并记录评估指标
                    info = infos[i]
                    metric = {}
                    metric['steps_taken'] = info['steps_taken']
                    ep_id = str(envs.current_episodes()[i].episode_id)
                    gt_path = np.array(self.gt_data[ep_id]['locations']).astype(float)  # 获取真实路径

                    # 获取智能体实际走过的路径和碰撞信息
                    if 'current_path' in envs.current_episodes()[i].info.keys():
                        positions_ = np.array(envs.current_episodes()[i].info['current_path']).astype(float)
                        collisions_ = np.array(envs.current_episodes()[i].info['collisions'])
                        assert collisions_.shape[0] == positions_.shape[0] - 1
                    else:
                        positions_ = np.array(dis_to_con(np.array(info['position']['position']))).astype(float)

                    distance = np.array(info['position']['distance']).astype(float) # 到目标的距离序列
                    metric['distance_to_goal'] = distance[-1]   # 最终距离
                    metric['success'] = 1. if distance[-1] <= 3. else 0.    # 是否成功（距离<3m
                    metric['oracle_success'] = 1. if (distance <= 3.).any() else 0. # 是否曾经成功过
                    metric['path_length'] = np.linalg.norm(positions_[1:] - positions_[:-1],axis=1).sum()   # 路径长度
                    metric['collisions'] = collisions_.mean()   # 平均碰撞率
                    gt_length = distance[0]
                    # 计算 SPL (Success weighted by Path Length)
                    metric['spl'] = metric['success']*gt_length/max(gt_length,metric['path_length'])

                    # 计算 nDTW (Normalized Dynamic Time Warping)
                    act_con_path = positions_
                    gt_con_path = np.array(gt_path).astype(float)
                    dtw_distance = fastdtw(act_con_path, gt_con_path, dist=NDTW.euclidean_distance)[0]
                    nDTW = np.exp(-dtw_distance / (len(gt_con_path) * config.TASK_CONFIG.TASK.SUCCESS_DISTANCE))

                    metric['ndtw'] = nDTW

                    # 将指标存储到字典中
                    stats_episodes[current_episodes[i].episode_id] = metric 

                    # 23.3 重置当前环境并获取新episode的初始观测
                    observations[i] = envs.reset_at(i)[0]
                    instruction, images_list = self.generate_input(observations[i])
                    
                    # 23.4 更新进度条或打印日志
                    if config.use_pbar:
                        pbar.update()
                    else:
                        logger.info(
                            log_str.format(
                                evaluated=len(stats_episodes),
                                total=episodes_to_eval,
                                time=round(time.time() - start_time),
                            )
                        )
                
                # 24. 为下一轮循环准备数据
                # 重新提取指令token，打包观测批次
                observations = extract_instruction_tokens(
                    observations,
                    self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID,
                )
                batch = batch_obs(observations, self.device)
                batch = apply_obs_transforms_batch(batch, obs_transforms)   
                
                # 25. 确定需要暂停的环境（已完成评估的
                envs_to_pause = []
                next_episodes = envs.current_episodes()

                for i in range(envs.num_envs):
                    if next_episodes[i].episode_id in stats_episodes:
                        envs_to_pause.append(i)

                # 26. 暂停已完成的环境
                headings = torch.tensor(headings)
                (
                    envs,
                    not_done_masks,
                    headings,  
                    batch,
                    rgb_frames,
                ) = self._pause_envs(
                    envs_to_pause,
                    envs,
                    not_done_masks,
                    headings,   # TODO：这里传入headings但返回值覆盖了它，可能是笔误，应该是prev_actions？
                    batch,
                    rgb_frames,
                )
                headings = headings.tolist()    # 更新headings列表
            
            # 27. 异常处理：如果在决策或执行过程中出错
            except Exception as e:
                nav_logger.info(f"Error in next action prediction: {e}")
                current_step -= 1
        
        # 28. 关闭环境和进度条
        envs.close()
        if config.use_pbar:
            pbar.close()

        # 29. 多GPU同步（如果使用分布式训练）
        if self.world_size > 1:
            distr.barrier()
        
        # 30. 聚合所有已完成episode的统计信息
        aggregated_stats = {}
        num_episodes = len(stats_episodes)
        # 计算每个指标的平均值
        for stat_key in next(iter(stats_episodes.values())).keys():
            aggregated_stats[stat_key] = (
                sum(v[stat_key] for v in stats_episodes.values())
                / num_episodes
            )

        # 31. 多GPU结果汇总
        total = torch.tensor(num_episodes).cuda()
        if self.world_size > 1:
            dist.reduce(total,dst=0)     # 将所有GPU的episode数汇总到GPU 0
        total = total.item()

        if self.world_size > 1:
            logger.info(
                f"rank {self.local_rank}'s {num_episodes}-episode results: {aggregated_stats}")
            for k,v in aggregated_stats.items():
                v = torch.tensor(v*num_episodes).cuda()
                cat_v = gather_list_and_concat(v,self.world_size)
                v = (sum(cat_v)/total).item()
                aggregated_stats[k] = v

        # 32. 保存结果到文件
        split = config.TASK_CONFIG.DATASET.SPLIT
        fname = os.path.join(
            config.RESULTS_DIR,
            f"stats_ep_ckpt_{split}_r{self.local_rank}_w{self.world_size}.json",
        )
        with open(fname, "w") as f:
            json.dump(stats_episodes, f, indent=4)

        # 33. 主进程（rank 0）保存聚合后的结果
        if self.local_rank < 1:
            if config.EVAL.SAVE_RESULTS:
                fname = os.path.join(
                    config.RESULTS_DIR,
                    f"stats_ckpt_{split}.json",
                )
                with open(fname, "w") as f:
                    json.dump(aggregated_stats, f, indent=4)

            # 34. 打印最终评估结果
            logger.info(f"Episodes evaluated: {total}")
            for k, v in aggregated_stats.items():
                logger.info(f"Average episode {k}: {v:.6f}")
        
    def collect_val_traj(self):
        trajectories = defaultdict(list)
        split = self.config.TASK_CONFIG.DATASET.SPLIT
        with gzip.open(
            self.config.TASK_CONFIG.TASK.NDTW.GT_PATH.format(
                split=split)
        ) as f:
            gt_data = json.load(f)
        self.gt_data = gt_data
        trajectories = gt_data
        self.trajectories = gt_data
        trajectories = list(trajectories.keys())[self.config.local_rank::self.config.GPU_NUMBERS]
        return trajectories
        
    def eval(self) -> None:
        r"""Main method of trainer evaluation. 

        Returns:
            None
        """
        self.device = (
            torch.device("cuda", self.config.TORCH_GPU_ID)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )

        if "tensorboard" in self.config.VIDEO_OPTION:
            assert (
                len(self.config.TENSORBOARD_DIR) > 0
            ), "Must specify a tensorboard directory for video display"
            os.makedirs(self.config.TENSORBOARD_DIR, exist_ok=True)
        if "disk" in self.config.VIDEO_OPTION:
            assert (
                len(self.config.VIDEO_DIR) > 0
            ), "Must specify a directory for storing videos on disk"

        world_size = self.config.GPU_NUMBERS
        self.world_size = world_size
        self.local_rank = self.config.local_rank

        self.config.defrost()
        self.config.TASK_CONFIG.DATASET.ROLES = ["guide"]
        self.config.TASK_CONFIG.TASK.MEASUREMENTS = ['POSITION',
                                                     'STEPS_TAKEN',
                                                     ]
        if 'HIGHTOLOW' in self.config.TASK_CONFIG.TASK.POSSIBLE_ACTIONS:
            idx = self.config.TASK_CONFIG.TASK.POSSIBLE_ACTIONS.index('HIGHTOLOW')
            self.config.TASK_CONFIG.TASK.POSSIBLE_ACTIONS[idx] = 'HIGHTOLOWEVAL'
        self.config.TASK_CONFIG.DATASET.LANGUAGES = self.config.EVAL.LANGUAGES
        self.config.TASK_CONFIG.DATASET.SPLIT = self.config.EVAL.SPLIT
        self.config.TASK_CONFIG.TASK.NDTW.SPLIT = self.config.EVAL.SPLIT
        self.config.TASK_CONFIG.TASK.SDTW.SPLIT = self.config.EVAL.SPLIT
        self.config.use_pbar = not is_slurm_batch_job()
        if 'rxr' in self.config.BASE_TASK_CONFIG_PATH:
            self.config.EVAL.trajectories_file = \
                self.config.EVAL.trajectories_file[:-8] + '_w' + \
                str(self.world_size) + '_r' + str(self.local_rank) + '.json.gz'
        
        # if choosing image
        resize_config = self.config.RL.POLICY.OBS_TRANSFORMS.RESIZER_PER_SENSOR.SIZES
        config = self.config.TASK_CONFIG
        camera_orientations = get_camera_orientations(12)

        # sensor_uuids = []
        for sensor_type in ["RGB", "DEPTH"]:
            resizer_size = dict(resize_config)[sensor_type.lower()]
            sensor = getattr(config.SIMULATOR, f"{sensor_type}_SENSOR")
            for action, orient in camera_orientations.items():
                camera_template = f"{sensor_type}_{action}"
                camera_config = deepcopy(sensor)
                camera_config.ORIENTATION = camera_orientations[action]
                camera_config.UUID = camera_template.lower()
                # sensor_uuids.append(camera_config.UUID)
                setattr(config.SIMULATOR, camera_template, camera_config)
                config.SIMULATOR.AGENT_0.SENSORS.append(camera_template)
                resize_config.append((camera_template.lower(), resizer_size))
        self.config.RL.POLICY.OBS_TRANSFORMS.RESIZER_PER_SENSOR.SIZES = resize_config
        self.config.TASK_CONFIG = config
        self.config.SENSORS = config.SIMULATOR.AGENT_0.SENSORS
        
        self.config.freeze()
        torch.cuda.set_device(self.device)
        if world_size > 1:
            distr.init_process_group(backend='nccl', init_method='env://')
            self.device = self.config.TORCH_GPU_IDS[self.local_rank]
            torch.cuda.set_device(self.device)
            self.config.defrost()
            self.config.TORCH_GPU_ID = self.config.TORCH_GPU_IDS[self.local_rank]
            self.config.freeze()
            
        self.traj = self.collect_val_traj()
        self._eval_llm()


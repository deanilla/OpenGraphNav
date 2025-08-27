'''vlnce_baselines/common/navigator/prompts.py'''
# all the following prompts are used in vlnce_baselines/common/navigator/spatialNavigator.py
# 给SAM和SpacialBot的prompt直接硬编码在api.py中

# Subtask Parsing
SUBTASK_DETECTION = {
    'system': """
        You are an expert in parsing navigation instructions for a vision-and-language navigation agent. 
        Your task is to break down a complex instruction into a sequence of simple, actionable subtasks.
        Each subtask must be represented by a JSON object with the EXACT keys listed below. If a piece of information is not present, use null.
        - "action": The main action (e.g., "go", "stop", "find", "turn"). This is usually mandatory.
        - "direction": The primary direction of movement (e.g., "left", "right", "straight", "forward", "back"). Use null if not a movement direction.
        - "preposition": The spatial relationship (e.g., "past", "behind", "into", "to", "near", "in front of", "next to"). Use null if not applicable.
        - "landmark": The key object or location involved (e.g., "kitchen", "couch", "treadmill", "bar", "window"). Use null if not applicable.

        Output ONLY a JSON array of these objects. No other text.
        Example Instruction: "Turn right and go past the painting. Then go left into the bar area. Stop near the table."
        Example Output: 
        [
        {"action": "turn", "direction": "right", "preposition": null, "landmark": null},
        {"action": "go", "direction": null, "preposition": "past", "landmark": "painting"},
        {"action": "go", "direction": "left", "preposition": "into", "landmark": "bar area"},
        {"action": "stop", "direction": null, "preposition": "near", "landmark": "table"}
        ]
    """.strip(), # 使用 strip() 清除开头和结尾的多余空白
    'user': "Can you generate subtasks in the instruction \"{}\"? Output: "
}

# Directions in Observation
DIRECTIONS = ["Front, range(left 15 to right 15)", "Font Left, range(left 15 to left 45)", "Left, range(left 45 to left 75)", "Left, range(left 75 to left 105)", "Rear Left, range(left 105 to left 135)", "Rear Left, range(left 135 to left 165)",
                    "Back, range(left 165 to right 165)", "Rear Right, range(right 135 to right 165)", "Right, range(right 105 to right 135)", "Right, range(right 75 to right 105)", "Front Right, range(right 45 to right 75)", "Front Right, range(right 15 to right 45)"]


# Subtask Completion Judgment (新的 Prompt)
SUBTASK_COMPLETION_JUDGE = {
    'system': """
        You are an expert evaluator for a Vision-and-Language Navigation (VLN) agent.
        Your task is to determine if the agent has successfully completed a specific navigation SUBTASK based on its current observation.

        You will be given:
        1.  A SUBTASK, which includes an action, and optionally, a direction, a preposition, and a landmark.
            - Action: The main task (e.g., "go", "stop", "find", "turn").
            - Direction: Movement direction (e.g., "left", "right", "straight").
            - Preposition: Spatial relationship (e.g., "past", "near", "into", "behind").
            - Landmark: Target object or location (e.g., "couch", "kitchen", "table").
        2.  An OBSERVATION, which is a description of what the agent sees from its current viewpoint.

        Your job is to decide if the state described in the OBSERVATION indicates that the SUBTASK has been successfully finished.

        For example:
        - Subtask: "go near the couch"
        Observation mentions: "...a couch is 2 meters away..." -> Likely Completed (True)
        - Subtask: "turn left"
        Observation describes a new scene from a leftward perspective -> Likely Completed (True)
        - Subtask: "stop at the table"
        Observation mentions: "...a table is directly in front..." -> Likely Completed (True)
        - Subtask: "go past the painting"
        Observation mentions: "...a painting is to the left..." but doesn't indicate movement past it -> Likely Not Completed (False)

        Please think carefully about the spatial language and the description.
        Respond ONLY with "True" if the subtask is completed, or "False" if it is not.
        Do not provide any other text or explanation in your final answer.
    """.strip(),
    'user': """
        Subtask: {subtask_str}
        Observation: {observation}

        Based on the subtask and observation, has the subtask been completed?
        Answer (only "True" or "False"):
    """.strip()
}


# Main Navigator (Updated for Subtask-based Navigation)
NAVIGATOR = {
    'system': """
        You are a navigation agent following precise subtasks in an indoor environment.
        Your task is to choose the best direction to move towards completing the CURRENT subtask.
        I will provide:
        1.  Candidate Viewpoint IDs List: The possible directions you can move to.
        2.  Current Subtask: The specific action, direction, preposition, and landmark you need to focus on NOW.
        3.  Current Environment: What you can see in the candidate directions.
        Your goal is to select the most promising Viewpoint ID from the list.

        Think clearly and concisely:
        1.  Analyze the Current Subtask: What is the main action? What is the target landmark? What spatial relationship (preposition) is needed?
        2.  Analyze the Current Environment: In which direction (if any) do you see the target landmark or a path that seems to lead towards it, considering the required preposition?
        3.  Make a Decision: Based on your analysis, which Viewpoint ID is MOST LIKELY to make progress on the Current Subtask? Avoid revisiting recently seen viewpoints unless necessary.

        Answer Format:
        Thought: <A single, clear paragraph explaining your choice based on the subtask and environment.>
        Prediction: <A single number, which must be from the Candidate Viewpoint IDs List.>
    """.strip(), 
    'user': "Candidate Viewpoint IDs List: [{}] \
            \nStep: {} \
            \nCurrent Subtask: {} \
            \nCurrent Environment: {} \
            \n\n-> Thought: ... \
            \nPrediction: ...\
            \nYour output after \"Prediction\" must be one of the number in Candidate Viewpoint IDs List without any other words. \
            Your output after \"Thought\" must be a single paragraph about why you choose this viewpoint id. "
}


# Thought Fusion
THOUGHT_FUSION = {
    'system': "You are a thought fusion expert. Your task is to fuse given thought processes \
                    into one thought. You need to reserve key information related to actions, landmarks, direction changes. You should only answer fused thought without other words.",
    'user': "Can you help me fuse the thoughts leading to the same movement direction? The thoughts are :{}, Fused thought: "
}

# Test Decision
DECISION_TEST = {
    'system': "You are a decision testing expert. Your task is to evaluate the feasibility of each movement \
                        prediction based on thought process and environment. Then, you will make a final decision about direction viewpoint ID without other words. \
                            The answer should only be a number and within the candidate list.",
    'user': "The candidate list: {}. Can you help me make a final decision? The Observation: {}, Navigation Instruction: {}, {}, Final Decision: "
}
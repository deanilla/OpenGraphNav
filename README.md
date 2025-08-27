# ?

### TODO

- [ ] Subtask queue, to determine progress and current step
- [ ] Trajectory tree, to record navigation history (for waypoint navigation)
- [ ] Scene Graph, to represent the scene
- [ ] backtracking & blacklist mechanism, used with Trajectory tree

行进中rolling构建Scene Graph，用CA-Nav的action-landmarks sequence监控当前subtask，用SG-Nav的方法（从Object Goal延伸到VLN）计算每个waypoint的probability。

At time t, we divide the scene graph Gt into subgraphs, each of which is determined by an object node with all its parent nodes and other directly connected object nodes. 
For each subgraph, we predict P sub , the similarity between the destination sub-graph and this subgraph after executing the current (action, landmarks) subtask. 
Then the probability Pif ro of the i-th frontier can be averaged by:


Instruction parsing

action, direction, preposition, landmark

。e.g. 1 (adopted from Open-Nav paper)
Instruction: Walk past the kitchen and go behind the couch and take a right into the fitness room. Stop next to the treadmill.

State Constraint Queue:
go, /, past, kitchen
go, /, behind, couch
go, right, into, fitness room
go, /, to, treadmill
stop

e.g. 2 (adopted from CA-Nav paper)
Instruction: Turn right and go past the painting. Then go left to go into the bar area. Pass the bar and go towards the dining table. Stop in the dining room near the table.

State Constraint Queue:
go, right, past, painting
go, left, into, bar area
go, /, past, bar
go, /, to, dining table
stop
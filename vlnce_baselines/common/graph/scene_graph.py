'''vlnce_baselines/common/graph/scene_graph.py'''

from enum import Enum
from dataclasses import dataclass, field
import networkx as nx  # 使用 NetworkX 库来处理图结构
from typing import Dict, Any, Optional, List, Set, Tuple 

class NodeType(Enum):
    """
    定义场景图中节点的类型。
    """
    OBJECT = "object"     # 物体节点 (e.g., chair, table)
    GROUP = "group"       # 组节点 (e.g., dining set, TV area)
    ROOM = "room"         # 房间节点 (e.g., living room, kitchen)
    WAYPOINT = "waypoint" # 航点节点 (来自 trajectory tree)

    def __str__(self):
        return self.value
    
class EdgeType(Enum):
    """
    定义场景图中边的类型。
    """
    AFFILIATION = "affiliation"  # 跨层级的归属关系 (e.g., Object -> Room, Group -> Room)
    SPATIAL = "spatial"          # 同层级的空间关系 (e.g., Object -> Object: 'near', 'left_of')
    FUNCTIONAL = "functional"    # 同层级的功能关系 (e.g., Group -> Group: 'opposite')

    def __str__(self):
        return self.value
    

@dataclass
class SceneNode:
    """
    场景图中的一个节点。
    """
    id: str                 # 节点唯一标识符 (e.g., "obj_1", "grp_2", "room_3", "wp_4")
    type: NodeType          # 节点类型
    attributes: Dict[str, Any] = field(default_factory=dict)
    # attributes 可以包含:
    # - 对于 OBJECT: {'category': 'chair', 'instance_id': 123, 'color': 'red', 'bbox': ...}
    # - 对于 GROUP: {'category': 'dining_set', 'member_ids': ['obj_1', 'obj_2', ...]}
    # - 对于 ROOM: {'category': 'living_room', 'instance_id': 456}
    # - 对于 WAYPOINT: {'position': [x, y, z], 'heading': radian}
    
    def __str__(self):
        return f"Node(id='{self.id}', type={self.type}, attributes={self.attributes})"
    
@dataclass
class SceneEdge:
    """
    场景图中连接两个节点的边。
    """
    source_id: str      # 起始节点ID
    target_id: str      # 目标节点ID
    relation: str       # 关系描述 (e.g., 'inside', 'left_of', 'near', 'opposite')
    type: EdgeType      # 边的类型
    confidence: float = 1.0 # 关系置信度 (可选)

    def __str__(self):
        return f"Edge({self.source_id} --[{self.relation}, {self.type}]--> {self.target_id}, conf={self.confidence})"
    


class SceneGraph:
    """
    表示一个层次化的 3D 场景图。
    使用 NetworkX 图来存储节点和边，便于进行图算法操作。
    """
    def __init__(self):
        # 使用 NetworkX 的 MultiDiGraph，因为它允许节点间存在多种类型的边（例如，空间和功能关系）
        # 但为了简化，我们先用 DiGraph，并将关系类型存储在边的属性中。
        # 如果需要同一对节点间多种关系，DiGraph 也可以通过边属性区分。
        self.graph = nx.DiGraph() # 或 nx.MultiDiGraph() 如果需要更复杂的关系
        
        # 为了快速查找，可以维护一个从 ID 到 Node 对象的字典
        # NetworkX 的节点可以是任意对象，但我们通常用字符串 ID 作为节点，
        # 然后将 SceneNode 对象作为节点的属性存储。
        # self.nodes: Dict[str, SceneNode] = {} 
        # self.edges: List[SceneEdge] = [] 
        # 上述两个可以由 NetworkX 的 graph.nodes(data=True) 和 graph.edges(data=True) 替代

    def add_node(self, node: SceneNode):
        """
        向场景图中添加一个节点。
        如果节点 ID 已存在，则更新其属性。
        """
        # NetworkX 的 add_node 会自动处理节点已存在的情况（更新属性）
        # 我们将 SceneNode 对象本身作为 'data' 属性存储
        self.graph.add_node(node.id, data=node)
        # self.nodes[node.id] = node # 如果不使用 NetworkX 属性存储

    def get_node(self, node_id: str) -> Optional[SceneNode]:
        """
        根据 ID 获取场景图中的节点。
        """
        if self.graph.has_node(node_id):
            # 从 NetworkX 节点属性中获取 SceneNode 对象
            return self.graph.nodes[node_id]['data']
        return None
        # return self.nodes.get(node_id) # 如果使用自己的字典存储

    def add_edge(self, edge: SceneEdge):
        """
        向场景图中添加一条边。
        """
        # NetworkX 的 add_edge 会自动处理边已存在的情况（更新属性）
        # 我们将 SceneEdge 的属性存储在 NetworkX 边的属性中
        self.graph.add_edge(
            edge.source_id, edge.target_id,
            relation=edge.relation,
            type=edge.type,
            confidence=edge.confidence
        )
        # self.edges.append(edge) # 如果使用自己的列表存储

    def get_edges(self) -> List[SceneEdge]:
        """
        获取场景图中的所有边。
        """
        edges = []
        for u, v, data in self.graph.edges(data=True):
            edge = SceneEdge(
                source_id=u,
                target_id=v,
                relation=data.get('relation'),
                type=data.get('type'),
                confidence=data.get('confidence', 1.0)
            )
            edges.append(edge)
        return edges
        # return self.edges # 如果使用自己的列表存储

    def get_subgraph_around_node(self, center_node_id: str, radius: int = 2) -> Optional['SceneGraph']:
        """
        提取以 center_node_id 为中心，跳数为 radius 的局部子图。

        Args:
            center_node_id (str): 中心节点的 ID。
            radius (int): 从中心节点出发的跳数。

        Returns:
            Optional[SceneGraph]: 新的 SceneGraph 对象，包含子图；如果中心节点不存在则返回 None。
        """
        if not self.graph.has_node(center_node_id):
            print(f"Warning: Node {center_node_id} not found in graph for subgraph extraction.")
            return None

        # 使用 NetworkX 的 ego_graph 函数提取子图
        # ego_graph(G, center_node, radius) 返回一个包含 center_node 及其 radius 跳内邻居的子图
        try:
            subgraph_nx = nx.ego_graph(self.graph, center_node_id, radius=radius)
            
            # 创建一个新的 SceneGraph 对象来容纳子图
            subgraph = SceneGraph()
            # 复制节点
            for node_id, node_data in subgraph_nx.nodes(data=True):
                # node_data 包含 'data' 键，其值是原始的 SceneNode 对象
                subgraph.add_node(node_data['data'])
            # 复制边
            for u, v, edge_data in subgraph_nx.edges(data=True):
                edge = SceneEdge(
                    source_id=u,
                    target_id=v,
                    relation=edge_data.get('relation'),
                    type=edge_data.get('type'),
                    confidence=edge_data.get('confidence', 1.0)
                )
                subgraph.add_edge(edge)
            
            return subgraph
        except nx.NetworkXError as e:
            print(f"Error extracting subgraph: {e}")
            return None

    def serialize_to_text(self, detail_level: str = "concise") -> str:
        # TODO：detail_level
        """
        将整个场景图或子图序列化为 LLM 友好的文本描述。

        Args:
            detail_level (str): 描述的详细程度 ("concise", "detailed")。

        Returns:
            str: 序列化后的文本。
        """
        if self.graph.number_of_nodes() == 0:
            return "The scene graph is empty."

        lines = []
        lines.append("Scene Graph Description:")
        
        # --- 序列化节点 ---
        lines.append("Nodes:")
        for node_id, node_data in self.graph.nodes(data=True):
            node: SceneNode = node_data['data']
            attr_str = ", ".join([f"{k}: {v}" for k, v in node.attributes.items()])
            lines.append(f"  - {node_id} (type: {node.type}, {attr_str})")

        # --- 序列化边 ---
        lines.append("Edges:")
        for u, v, edge_data in self.graph.edges(data=True):
            relation = edge_data.get('relation', 'N/A')
            edge_type = edge_data.get('type', 'N/A')
            conf = edge_data.get('confidence', 1.0)
            lines.append(f"  - {u} --[{relation}, type: {edge_type}, conf: {conf:.2f}]--> {v}")
            
        return "\n".join(lines)

    def __str__(self):
        """
        提供一个 SceneGraph 的简洁字符串表示。
        """
        return f"SceneGraph(nodes={self.graph.number_of_nodes()}, edges={self.graph.number_of_edges()})"

# --- 辅助函数 ---

def nodes_to_text(nodes: List[SceneNode]) -> str:
    """将节点列表转换为文本描述。"""
    if not nodes:
        return "No nodes."
    lines = ["Nodes:"]
    for n in nodes:
        lines.append(f"  {n}")
    return "\n".join(lines)

def edges_to_text(edges: List[SceneEdge]) -> str:
    """将边列表转换为文本描述。"""
    if not edges:
        return "No edges."
    lines = ["Edges:"]
    for e in edges:
        lines.append(f"  {e}")
    return "\n".join(lines)

import numpy as np
from typing import Union
import torch
import json
import os


class SubNode:
    def __init__(
        self, sub_node_id: int, position: list | tuple, frame_id: int | str
    ) -> None:
        self.id = sub_node_id
        self.position = position
        self.frame_id = frame_id
        self.parent_node = None
        # the
        self.visible_in_frame = []

    def __str__(self) -> str:
        if self.parent_node is None:
            return f"SubNode {self.id} at {self.position} in frame {self.frame_id} with no parent node"
        return f"SubNode {self.id} at {self.position} in frame {self.frame_id} with parent node {self.parent_node}"

    def set_parent_node(self, node_id) -> None:
        self.parent_node = node_id


class Node:
    def __init__(self, node_id: int, position: list | tuple) -> None:
        self.id = node_id
        self.position = position
        self.subnodes = dict()

    def add_subnode(self, subnode: SubNode) -> None:
        self.subnodes[subnode.id] = subnode

    def get_subnode(self, subnode_id: int) -> SubNode:
        return self.subnodes[subnode_id]

    def __str__(self) -> str:
        sub_info = ""
        if len(self.subnodes) == 0:
            sub_info = "None"
        for _, subnode in self.subnodes.items():
            sub_info += str(subnode) + "\n"
        return (
            f"Node {self.id} at {self.position} with subnodes:" + "\n" + f"{sub_info}"
        )


class ViewGraph:
    def __init__(self) -> None:
        # list of node id: the node is the 3d points list
        self.nodes = dict()
        # list of subnodes id: the subnode is the 2d correspondences
        self.sub_nodes = dict()
        # the edge is the 2d correspondences between the 3d points and the frame, a list of tuple [(node_id,sub_node_id),...]
        self.view_edges = []

        # initialize the node and subnode id
        self.node_id = -1
        self.subnode_id = -1

    # create a signle node in view graph, 3d points = node
    def create_node(self, position: Union[list, np.array]) -> Node:
        # check position type
        if isinstance(position, list):
            position = np.array(position)
        # check if the node is already in the view graph
        node = self._node_exists(position)
        if node is None:
            node = Node(self._get_node_id(), position)
            self.nodes[node.id] = node
        return node

    # create multiple nodes in view graph
    def create_multi_nodes(self, positions: list) -> list:
        return [self.create_node(pos) for pos in positions]

    # create a single subnode in view graph, 2d correspondences = subnode
    def create_subnode(
        self, position: Union[list, np.array, torch.tensor], frame_id: int
    ) -> SubNode:

        # check if the subnode is already in the view graph
        sub_node = self._subnode_exists(position, frame_id)
        if sub_node is None:
            sub_node = SubNode(self._get_subnode_id(), position, frame_id)
            self.sub_nodes[sub_node.id] = sub_node

        return sub_node

    # add an edge between a node and a subnode
    def add_edge(
        self,
        node_id: int,
        subnode_id: int,
    ) -> None:
        # check if the node and subnode are already in the view graph
        assert (
            node_id in list(self.nodes.keys()),
            "The node is not in the view graph",
        )
        assert (
            subnode_id in list(self.sub_nodes.keys()),
            "The subnode is not in the view graph",
        )
        # check if the subnode belongs to the node
        check = self.frame_exists_in_node(self.sub_nodes[subnode_id].frame_id, node_id)
        if self.sub_nodes[subnode_id].parent_node is None and check == 0:
            self.nodes[node_id].add_subnode(self.sub_nodes[subnode_id])
            self.sub_nodes[subnode_id].set_parent_node(node_id)

    # add multiple edges between multiple pairs of nodes and subnodes
    def add_multi_edges(
        self,
        node_ids: list,
        subnode_ids: list,
    ) -> None:
        assert (
            len(node_ids) == len(subnode_ids),
            "The length of the node_ids, subnode_pos should be the same",
        )
        for node_id, subnode_id in zip(node_ids, subnode_ids):
            self.add_edge(node_id, subnode_id)

    def remove_node(self, node_id: int | str) -> int:
        if node_id in self.nodes.keys():
            for subnode in self.nodes[node_id].subnodes.values():
                del self.sub_nodes[subnode.id]
            del self.nodes[node_id]
            return 1
        else:
            return 0

    # search subnodes by frame id and return the node id and subnode id
    def search_nodes_by_frame(self, frame_id: int | str) -> list:
        node_pairs = []
        for node in self.nodes.values():
            for subnode in node.subnodes.values():
                if int(subnode.frame_id) == int(frame_id):
                    node_pairs.append((node.id, subnode.id))
        return node_pairs

    def frame_exists_in_node(self, frame_id: int | str, node_id: int) -> int:
        check = frame_id in [
            subnode.frame_id for subnode in self.nodes[node_id].subnodes.values()
        ]
        return int(check)

    # check if the subnode belongs to the node return the int value
    def _sub_node_belongs_to_node(self, subnode_id: int, node_id: int) -> int:
        check = subnode_id in self.nodes[node_id].subnodes.keys()
        return int(check)

    def _subnode_exists(
        self,
        subnode_pos: Union[list, np.array, torch.tensor],
        subnode_frame: str | int,
    ) -> SubNode | None:
        # check position type
        if type(subnode_pos) == np.array or type(subnode_pos) == list:
            subnode_pos = torch.tensor(subnode_pos)
        for subnode in self.sub_nodes.values():
            if int(subnode.frame_id) == int(subnode_frame):
                if (abs(subnode.position - subnode_pos) < 5).all():
                    return subnode
        return None

    def _node_exists(self, node_pos) -> None | Node:
        if type(node_pos) == np.array or type(node_pos) == torch.Tensor:
            node_pos = torch.tensor(node_pos)
        for node in self.nodes.values():
            if (node.position == node_pos).all():
                return node
        return None

    def remove_subnode(self, node_id: int) -> None:
        del self.sub_nodes[node_id]

    def _get_nodes_len(self) -> int:
        return len(self.nodes)

    def _get_subnodes_len(self) -> int:
        return len(self.sub_nodes)

    def _get_node_id(self) -> int:
        self.node_id += 1
        return self.node_id

    def _get_subnode_id(self) -> int:
        self.subnode_id += 1
        return self.subnode_id

    def _iter_nodes(self):
        print("----iterate nodes----")
        for node in self.nodes.values():
            print(node)
        print("----end of nodes----")

    def _iter_subnodes(self):
        print("----iterate subnodes----")
        for subnode in self.sub_nodes.values():
            print(subnode)
        print("----end of subnodes----")

    def _toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__)

    def save_to_json(self, path):
        file_path = os.path.join("./estimation", path)
        if not os.path.isfile(file_path):
            os.mknod(file_path)

        data = {
            "nodes": {
                node_id: {
                    "position": node.position.tolist(),
                    "subnodes": {
                        subnode_id: {
                            "position": subnode.position.tolist(),
                            "frame_id": subnode.frame_id,
                        }
                        for subnode_id, subnode in node.subnodes.items()
                    },
                }
                for node_id, node in self.nodes.items()
            },
            "subnodes": {
                subnode_id: {
                    "position": subnode.position.tolist(),
                    "frame_id": subnode.frame_id,
                    "parent_node": subnode.parent_node,
                }
                for subnode_id, subnode in self.sub_nodes.items()
            },
        }
        with open(file_path, "w") as f:
            json.dump(data, f)


if __name__ == "__main__":

    # test the view graph
    view_graph = ViewGraph()

    # add a node to the view graph
    node0 = view_graph.create_node(torch.tensor([1, 1, 1]))
    # add a subnode to the view graph
    view_graph.add_edge(node0.id, view_graph.create_subnode(torch.tensor([1, 1]), 5).id)
    view_graph.add_edge(node0.id, view_graph.create_subnode(torch.tensor([2, 1]), 4).id)
    view_graph.add_edge(node0.id, view_graph.create_subnode(torch.tensor([2, 3]), 7).id)

    print("Adding node 0", view_graph.nodes[node0.id])

    # add another node to the view graph
    node1 = view_graph.create_node(torch.tensor([2, 1, 0]))
    view_graph.add_edge(node1.id, view_graph.create_subnode(torch.tensor([1, 1]), 9).id)
    view_graph.add_edge(
        node1.id, view_graph.create_subnode(torch.tensor([2, 1]), 41).id
    )
    view_graph.add_edge(
        node1.id, view_graph.create_subnode(torch.tensor([2, 3]), 17).id
    )

    print("Adding node 1", view_graph.nodes[node1.id])

    # add a subnode to the view graph
    node2 = view_graph.create_node(torch.tensor([2, 4, 0]))
    view_graph.add_edge(node2.id, view_graph.create_subnode(torch.tensor([1, 9]), 9).id)
    view_graph.add_edge(
        node2.id, view_graph.create_subnode(torch.tensor([2, 9]), 41).id
    )
    print("Adding node 2", view_graph.nodes[node2.id])
    view_graph._iter_nodes()
    view_graph._iter_subnodes()

    # delete a node from the view graph
    view_graph.remove_node(node2.id)
    print("After removing node 2")
    view_graph._iter_nodes()
    view_graph._iter_subnodes()

    # add new subnodes to existing nodes
    view_graph.add_edge(node0.id, view_graph.create_subnode(torch.tensor([4, 1]), 6).id)
    print("Adding subnode to node 0", view_graph.nodes[node0.id])
    view_graph.add_edge(
        node1.id, view_graph.create_subnode(torch.tensor([4, 1]), 10).id
    )
    print("Adding subnode to node 1", view_graph.nodes[node1.id])

    # create mutliple nodes and subnodes
    multi_nodes = view_graph.create_multi_nodes(
        torch.tensor([[5, 1, 1], [5, 1, 0], [5, 4, 0], [2, 1, 0]])
    )
    print("Creating multiple nodes", [str(node) for node in multi_nodes])

    multi_subnodes = [
        view_graph.create_subnode(torch.tensor([5, 1]), 5),
        view_graph.create_subnode(torch.tensor([5, 2]), 6),
        view_graph.create_subnode(torch.tensor([5, 3]), 7),
        view_graph.create_subnode(torch.tensor([6, 4]), 8),
    ]

    # add multiple subnodes to the view graph
    # for node, subnode in zip(multi_nodes, multi_subnodes):
    #     view_graph.add_edge(node.id, subnode.position, subnode.frame_id)

    # add multiple edges to the view graph
    view_graph.add_multi_edges(
        [node.id for node in multi_nodes],
        [subnode.id for subnode in multi_subnodes],
    )

    print("Adding multiple subnodes")
    view_graph._iter_nodes()

    # find the matched subnode
    next_subnode = view_graph.create_subnode(torch.tensor([5, 1]), 5)
    matches_pairs = [[5, 4], [5, 9], [5, 41], [5, 17]]
    for subnode in view_graph.sub_nodes.values():
        if ([subnode.frame_id, next_subnode.frame_id] in matches_pairs) or (
            [next_subnode.frame_id, subnode.frame_id] in matches_pairs
        ):
            print("Matched subnode", matches_pairs)

    # add repeated subnodes to the view graph
    print("Adding repeated subnodes")
    repeat_subnode1 = view_graph.create_subnode(torch.tensor([5, 1]), 5)
    view_graph.add_edge(node0.id, repeat_subnode1.id)
    print("Adding repeated subnode to node 0", view_graph.nodes[node0.id])

    # add repeated nodes to the view graph
    view_graph.create_node(torch.tensor([2, 1, 1]))
    view_graph._iter_nodes()
    view_graph._iter_subnodes()

    # search nodes by frame id
    print("Searching nodes by frame id")
    node_pairs = view_graph.search_nodes_by_frame(6)
    for pair in node_pairs:
        node_id, subnode_id = pair
        print(
            str(view_graph.nodes[node_id].position),
            str(view_graph.sub_nodes[subnode_id]),
        )

    # save the view graph to a json file
    print("Saving the view graph to a json file")
    view_graph.save_to_json("view_graph.json")

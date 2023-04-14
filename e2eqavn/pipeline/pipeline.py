import logging
from typing import *
from abc import abstractmethod
import networkx
from networkx import DiGraph
from networkx.drawing.nx_agraph import to_agraph


class StartNode:
    outgoing_edges = 1

    def run(self, **kwargs):
        return kwargs


class Pipeline:
    def __init__(self, root_id: str = 'root'):
        self.graph = DiGraph()
        self.root_id = root_id
        self.graph.add_node(node_for_adding=self.root_id, component=StartNode())

    def add_node(self, component, name_component: str, input_component: str):
        self.graph.add_node(
            node_for_adding=name_component,
            component=component,
            input_component=input_component
        )
        self.graph.add_edge(
            u_of_edge=input_component,
            v_of_edge=name_component
        )

    def run(self, **kwargs):
        queue = {
            self.root_id: {**kwargs}
        }
        i = 0
        while queue:
            current_node_id = list(queue.keys())[i]
            node_input = queue[current_node_id]
            output_node = self.graph.nodes[current_node_id]['component'].run(
                **node_input
            )
            # if "documents" in output_node:
            #     for i in range(len(output_node['documents'])):
            #         for doc in output_node['documents'][i]:
            #             print(doc.document_id, doc.embedding_similarity_score)
            queue.pop(current_node_id)
            current_node_id = self.get_next_node(current_node_id)
            if current_node_id is None:
                break
            queue[current_node_id] = output_node
        return output_node

    def draw(self, path_save: str):
        graph_picture = to_agraph(self.graph)
        graph_picture.layout("dot")
        graph_picture.draw(path_save)

    def get_next_node(self, node_id: str):
        current_edges = self.graph.edges(node_id)
        next_nodes = [
            next_node for _, next_node in current_edges
        ]
        try:
            return next_nodes[0]
        except:
            return None

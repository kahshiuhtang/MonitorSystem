import os
from DetectorLayer import DetectorLayer
from LayerGenerator import LayerGenerator
from Detector import Detector
import networkx as nx
import matplotlib.pyplot as plt
from networkx.readwrite import json_graph
import json


class HierarchalManager:
    def __init__(self, top_layer_nodes=2):
        self.mTop_layer_size_min = top_layer_nodes
        self.mLayers = []
        self.mNext_available_id = 0
        self.mGenerated_graph = None

    def load_hierarchy(self, graph_filepath):
        graph_data = None
        with open(graph_filepath, 'r') as f:
            graph_data = json.load(f)
            nodes = graph_data['nodes']
            total_nodes = []
            for node in nodes:
                total_nodes.append(Detector(node['id'], "", -1, []))
            edges = graph_data['edges']

        return

    def create_base_layer(self, path=None, files=None):
        if path is not None:
            starting_layer = DetectorLayer(1)
            directory = os.path.join("c:\\", "path")
            for root, dirs, files in os.walk(directory):
                for file in files:
                    if file.endswith(".csv"):
                        starting_layer.add_detector(
                            file, self.mNext_available_id)
                        self.mNext_available_id += 1
            self.mLayers.append(starting_layer)
            print("Successfully loaded first layer of data.")
            self.mNext_available_id += 1
        elif files is not None:
            starting_layer = DetectorLayer(1)
            for file in files:
                starting_layer.add_detector(
                    "low-level", self.mNext_available_id, file_name=file, lower_level_detectors=[])
                self.mNext_available_id += 1
            print("Successfully loaded first layer of data.")
            self.mLayers.append(starting_layer)
            self.mNext_available_id += 1
            return True
        else:
            print("[Error]: No data")
            return False
        return

    def create_higher_level_layers(self):
        if len(self.mLayers) > 1:
            print("Cluster & Group has already finished running")
            return False
        while True:
            highest_layer = self.mLayers[len(self.mLayers) - 1]
            gen = LayerGenerator(highest_layer, self.mNext_available_id)
            new_layer, next_avail_id = gen.create_new_layer(len(self.mLayers))
            self.mNext_available_id = next_avail_id
            self.mLayers.append(new_layer)
            if len(new_layer.mDetector_map) <= self.mTop_layer_size_min:
                break
        return True

    def save_graph_structure(self):
        print("Started saving detector hierarchy")
        if self.mGenerated_graph is None:
            print("Cannot save graph as it is None")
            return False
        if os.path.exists("hierarchy") == False:
            os.makedirs("hierarchy")
        # Serializing json
        graph = nx.node_link_data(self.mGenerated_graph)
        graph.update({"layers": len(self.mLayers)})
        graph_json_object = json.dumps(
            graph, indent=4)
        with open("hierarchy/hierarchy_graph.json", "w") as outfile:
            outfile.write(graph_json_object)
        print("Saved hierarchy into file system")
        return True

    def save_higher_level_detector_data(self):
        if os.path.exists("hierarchy") == False:
            os.makedirs("hierarchy")
        for i in range(0, len(self.mLayers)):
            if i != 0:
                self.mLayers[i].save()
        print("Finished saving all generated detector data")
        return

    def display_structure(self):
        layer_num = 0
        G = nx.DiGraph()
        for layer in self.mLayers:
            for id in layer.mDetector_map:
                G.add_node(id)
        for layer in self.mLayers:
            if layer_num != 0:
                for id in layer.mDetector_map:
                    detector = layer.mDetector_map[id]
                    for lower_level_detect_id in detector.mLowerLevelDetectorIDs:
                        G.add_edge(detector.mID, lower_level_detect_id)
            layer_num += 1
        pos = nx.planar_layout(G)
        self.mGenerated_graph = G
        # node_options = {"node_color": "blue",
        #                "node_size": 200}
        # edge_options = {"width": .50, "alpha": .5, "edge_color": "black"}
        # nx.draw_networkx_nodes(G, pos, **node_options)
        # nx.draw_networkx_edges(G, pos, **edge_options)
        nx.draw(G, pos=pos, with_labels=True)
        plt.show()
        layer_num = 0
        prev_size = 0
        points = dict()
        for layer in self.mLayers:
            if layer_num != 0:
                counted = 0
                x_size = prev_size / (2 + len(layer.mDetector_map))
                for id in layer.mDetector_map:
                    detector = layer.mDetector_map[id]
                    counted += 1
                    plt.plot(counted * x_size, layer_num)
                    plt.text(counted * x_size, layer_num, id, fontsize=9)
                    points.update({id: [counted * x_size, layer_num]})
                    for lower_level_detect_id in detector.mLowerLevelDetectorIDs:
                        x = [points[lower_level_detect_id][0], counted * x_size]
                        y = [layer_num - 1, layer_num]
                        plt.plot(x, y, marker='o')
            else:
                prev_size = len(layer.mDetector_map)
                for id in layer.mDetector_map:
                    detector = layer.mDetector_map[id]
                    plt.plot(id, 0)
                    plt.text(id, layer_num, id, fontsize=9)
                    points.update({id: [id, 0]})
            layer_num += 1
        plt.show()
        return True

from Detector import Detector
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import os


class DetectorLayer:
    def __init__(self, layer_number):
        self.MPL_COLORS = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
        self.mDetector_map = dict()
        self.mCurrentTime = 1
        self.mLayerNumber = layer_number

    # Add a detector that is being watched
    def add_detector(self, detector_type, detector_id, file_name=None, loaded_data=None, interval=None, lower_level_detectors=None):
        temp_detector = Detector(
            detector_id, detector_type, self.mLayerNumber, lower_level_detectors)
        if file_name is not None:
            temp_detector.load_from_file(file_name)
        if loaded_data is not None:
            temp_detector.load_from_memory(interval, loaded_data)
        self.mDetector_map.update({int(detector_id): temp_detector})
        return True

    def add_detect(self, detector):
        self.mDetector_map[str(detector.mID)] = detector
        return True

    # Remove one specific detector based on ID
    # Returns TRUE if detector was found, otherwise FALSE
    def delete_detector(self, id):
        if id in self.mDetector_map:
            del self.mDetector_map[id]
            return True
        return False

    # Graph multiple detector table on one graph
    def graph_by_ids(self, ids, x_range=None):
        if len(ids) > len(self.MPL_COLORS):
            print("Max lines on graph is " + str(len(self.MPL_COLORS)))
            return
        counter = 0
        for id in ids:
            if id in self.mDetector_map:
                x_data, y_data = self.mDetector_map[id].create_data(x_range
                                                                    )
                plt.plot(x_data, y_data, c=self.MPL_COLORS[counter])
                for anomaly in anomaly_data:
                    plt.plot(anomaly[0], anomaly[1], marker="x", markersize=5,
                             markeredgecolor='black', markerfacecolor='black')
                counter += 1
        plt.xlabel('Timestamp')
        plt.ylabel('Value')
        plt.title('Values over time')
        plt.show()
        return

    def generate_detector_data_array(self):
        answer = []
        for detector in self.mDetector_map:
            _, y_data = self.mDetector_map[detector].create_data()
            answer.append(y_data[:800])
        return np.transpose(answer)

    def save(self):
        for detector_id in self.mDetector_map:
            if os.path.exists("hierarchy/layer"+str(self.mLayerNumber)) == False:
                os.makedirs("hierarchy/layer"+str(self.mLayerNumber))
            self.mDetector_map[detector_id].save_history(
                "hierarchy/layer"+str(self.mLayerNumber)+"/detector"+str(detector_id)+"_id.csv")
        return True

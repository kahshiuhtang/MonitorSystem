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

    # Calls the next() method in all of the Detector objects being tracked
    def next(self):
        print("Current Time::" + str(self.mCurrentTime))
        for detector in self.mDetector_map:
            self.mDetector_map[detector].next(self.mCurrentTime)
        print("------------------------------------")
        self.mCurrentTime += 1
        return

    # Add a detector that is being watched
    def add_detector(self, detector_type, detector_id, file_name=None, loaded_data=None, interval=None, lower_level_detectors=None):
        temp_detector = Detector(
            detector_id, detector_type, self.mLayerNumber, lower_level_detectors)
        if file_name is not None:
            temp_detector.load_from_file(file_name)
        if loaded_data is not None:
            temp_detector.load_from_memory(interval, loaded_data)
        self.mDetector_map.update({detector_id: temp_detector})
        if detector_id == 69:
            temp_detector.graph_data()
        return True

    # Remove one specific detector based on ID
    # Returns TRUE if detector was found, otherwise FALSE
    def delete_detector(self, id):
        if id in self.mDetector_map:
            del self.mDetector_map[id]
            return True
        return False

    # Calls next() function repeatedly for duration of timestamp, works on all detectors
    def start(self, start_timestamp, end_timestamp):
        if end_timestamp < start_timestamp:
            print("Cannot end timestamp before it starts")
            return
        self.mCurrentTime = start_timestamp
        for i in range(start_timestamp, end_timestamp):
            self.next()
        return

    # Graph on detector data
    def graph_by_id(self, id, x_range):
        if id in self.mDetector_map:
            self.mDetector_map[id].graph_data(x_range)
            return True
        return False

    # Graph the anomaly data for a single detector by ID
    def graph_anomaly_by_id(self, id, x_range=None):
        if id in self.mDetector_map:
            self.mDetector_map[id].graph_anomaly(x_range)
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
                x_data, y_data, anomaly_data = self.mDetector_map[id].create_data(x_range
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
            _, y_data, _ = self.mDetector_map[detector].create_data()
            answer.append(y_data[:700])
        return np.transpose(np.array(answer))

    def save(self):
        for detector_id in self.mDetector_map:
            if os.path.exists("hierarchy/layer"+str(self.mLayerNumber)) == False:
                os.makedirs("hierarchy/layer"+str(self.mLayerNumber))
            self.mDetector_map[detector_id].save_history(
                "hierarchy/layer"+str(self.mLayerNumber)+"/detector"+str(detector_id)+"_id.csv")
        return True

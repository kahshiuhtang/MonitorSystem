from Detector import Detector
import matplotlib.pyplot as plt
import numpy as np


class DetectorManager:
    def __init__(self):
        self.MPL_COLORS = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
        self.mDetector_map = dict()
        self.mNext_available_id = 0
        self.mCurrentTime = 1

    # Calls the next() method in all of the Detector objects being tracked
    def next(self):
        print("Current Time::" + str(self.mCurrentTime))
        for detector in self.mDetector_map:
            self.mDetector_map[detector].next(self.mCurrentTime)
        print("------------------------------------")
        self.mCurrentTime += 1
        return

    # Add a detector that is being watched
    def add_detector(self, detector_type, file_name):
        temp_detector = Detector(self.mNext_available_id, detector_type, 1)
        if file_name is not None:
            temp_detector.load_from_file(file_name)
        self.mDetector_map.update({self.mNext_available_id: temp_detector})
        self.mNext_available_id += 1
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

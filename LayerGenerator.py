from Detector import Detector
from DetectorLayer import DetectorLayer
from matplotlib import pyplot as plt

import numpy as np
from sklearn.covariance import GraphicalLasso
from sklearn.cluster import AffinityPropagation
from sklearn import preprocessing
from sklearn import linear_model


class LayerGenerator:

    def __init__(self, layer, next_avail_id):
        self.mManager = layer
        self.mNew_detector_list = []
        self.mNext_available_id = next_avail_id

    def get_interval(self):
        ans = []
        if self.mManager:
            ans, _, _ = list(self.mManager.mDetector_map.values())[
                0].create_data()
        return ans

    def group(self):
        X = self.mManager.generate_detector_data_array()
        scaler = preprocessing.StandardScaler().fit(X)
        X_scaled = scaler.transform(X)
        cov = GraphicalLasso(alpha=0.05,
                             max_iter=2000).fit(X_scaled)
        print("Finished grouping layer")
        return np.around(cov.covariance_, decimals=3)

    def cluster(self, X):
        clustering = AffinityPropagation(random_state=5).fit_predict(X)
        detector_list = list(self.mManager.mDetector_map.values())
        detector_sets = dict()  # Keys are the new IDs and Values are the old detector values
        for idx in range(0, len(clustering)):
            if clustering[idx] + self.mNext_available_id in detector_sets:
                detector_sets[clustering[idx] + self.mNext_available_id].append(
                    detector_list[idx].mID)
            else:
                detector_sets.update(
                    {clustering[idx] + self.mNext_available_id: [detector_list[idx].mID]})
        print("Finished clustering layer")
        self.graph(detector_sets)
        # self.generate_new_detector_classifiers(detector_sets)
        self.generate_new_detector_means(detector_sets)
        return detector_sets

    def group_and_cluster(self):
        return self.cluster(self.group())

    def generate_new_detector_classifiers(self, detector_sets):
        summed_data = []
        for idx_array in detector_sets:
            temp_array = []
            for idx in detector_sets[idx_array]:
                detector = self.mManager.mDetector_map[idx]
                _, data, _ = detector.create_data()
                temp_array.append(data[0:700])  # TEMP fix
            summed_data.append(temp_array)
        for data_group in summed_data:
            clf = linear_model.SGDOneClassSVM(random_state=42)
            clf.fit(data_group)
            self.mNew_detector_list.append(clf)
        return

    def generate_new_detector_means(self, detector_sets):
        print("Started summing detector means for higher-level detectors")
        summed_data = []
        lower_level_detector_ids = []
        for idx_array in detector_sets:
            temp_array = []
            lower_level_detector_ids.append(detector_sets[idx_array])
            for idx in detector_sets[idx_array]:
                detector = self.mManager.mDetector_map[idx]
                _, data, _ = detector.create_data()
                temp_array.append(data[0:700])
            meaned_data = np.mean(np.array(temp_array), axis=0)
            summed_data.append(meaned_data)
        print("Finished summing detector means for higher-level detectors")
        return summed_data, lower_level_detector_ids

    def graph(self, clusters):
        groups = []
        data = []
        for item in clusters:
            total_str = ""
            for detector in clusters[item]:
                total_str += str(detector) + " "
            groups.append("Group " + str(item) +
                          ": Detectors(" + total_str + ")")
            data.append(len(clusters[item]))
        fig = plt.figure(figsize=(10, 7))
        plt.pie(data, labels=groups)
        plt.show()
        return

    def create_new_layer(self, layer_level):
        print("Started creating new layer, " + str(layer_level))
        detector_groups = self.group_and_cluster()
        interval = self.get_interval()
        new_data, lower_level_ids = self.generate_new_detector_means(
            detector_groups)
        layer = DetectorLayer(layer_level)
        idx = 0
        for detector_group_data in new_data:
            layer.add_detector("mid-level", self.mNext_available_id,
                               loaded_data=detector_group_data, interval=interval, lower_level_detectors=lower_level_ids[idx])
            self.mNext_available_id += 1
            idx += 1
        print("Finished creating new layer, " + str(layer_level))
        return layer, self.mNext_available_id

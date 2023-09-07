from Detector import Detector
from DetectorManager import DetectorManager

import numpy as np
from sklearn.covariance import GraphicalLasso
from sklearn.cluster import AffinityPropagation
from sklearn import preprocessing


class HierarchalDetector:

    def __init__(self, detector_manager):
        self.manager = detector_manager
        self.layers = None

    def group(self):
        X = self.manager.generate_detector_data_array()
        scaler = preprocessing.StandardScaler().fit(X)
        X_scaled = scaler.transform(X)
        cov = GraphicalLasso(alpha=0.05,
                             max_iter=2000).fit(X_scaled)
        return np.around(cov.covariance_, decimals=3)

    def cluster(self, X):
        clustering = AffinityPropagation(random_state=5).fit_predict(X)
        print(clustering)
        detector_sets = dict()
        for idx in range(0, len(clustering)):
            if clustering[idx] in detector_sets:
                detector_sets[clustering[idx]].append(idx)
            else:
                detector_sets.update({clustering[idx]: [idx]})
        return detector_sets

    def group_and_cluster(self):
        return self.cluster(self.group())

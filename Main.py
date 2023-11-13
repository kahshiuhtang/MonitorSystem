import sys
import os
from HierarchalManager import HierarchalManager
import numpy as np
# Steps
# 1) Data should be stored for 15 days
# 2) Service will run every 15 minutes
# 3) Takes most recent data form past 12 hours
# 4) Distributed Analysis applies learned detectors on data
# 5) Send results -> send alert decisions to CAS
# 6) CAS sends notifications to stakeholders

# 1) Read in the csv files
# 2) Put into detectors
# 3) Group & Cluster
# 4) Check display any anomalies
# 5) Display any anomalies

hManager = HierarchalManager()
files = []

path = r'test-pci'
extension = '.csv'

for root, dirs_list, files_list in os.walk(path):
    for file_name in files_list:
        if os.path.splitext(file_name)[-1] == extension:
            file_name_path = os.path.join(root, file_name)
            files.append("test-pci/" + file_name)
hManager.create_base_layer(files=files)
hManager.create_higher_level_layers()
"""
hManager.display_structure2()
hManager.save_graph_structure()
hManager.save_higher_level_detector_data()
print(hManager.find_order())
"""
"""
# Loading data
# Load in graph structure
hManager = HierarchalManager()
hManager.load_hierarchy("hierarchy/hierarchy_graph.json")
"""

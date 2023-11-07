from Detector import Detector
from DetectorLayer import DetectorLayer
from LayerGenerator import LayerGenerator
from HierarchalManager import HierarchalManager
import os


def load_mult_detector_graph():
    # Loading in all the data form A11Benchmark into detector classes
    manager = DetectorLayer(1)
    for i in range(1, 68):
        manager.add_detector("low_level_detector",
                             "data/A1Benchmark/real_"+str(i)+".csv")

    manager.start(1, 500)
    manager.graph_by_ids([1, 2, 3, 4], [1350, 1400])


def load_single_detector_graph():
    # Test the get_value_from_timestamp function
    temp = Detector(2, "low_level_detector", 1)
    temp.load_from_file("data/A1Benchmark/real_"+str(4)+".csv")
    temp.graph_data()
    print(temp.get_value_from_timestamp(12, "is_anomaly"))


def load_single_detector_graph_anomaly(x_range=None):
    # Test the get_value_from_timestamp function
    temp = Detector(3, "low_level_detector", 1)
    temp.load_from_file("data/A1Benchmark/real_"+str(4)+".csv")
    temp.graph_anomaly(x_range)


def load_single_detector_find_anomalies():
    # Test the get_value_from_timestamp function
    temp = Detector(2, "low_level_detector", 1)
    temp.load_from_file("data/A1Benchmark/real_"+str(4)+".csv")
    print(temp.find_anomaly(0.05))


def rho_estimation(rho_value_to_test):
    temp = Detector(2, "low_level_detector", 1)
    temp.load_from_file("data/A1Benchmark/real_"+str(5)+".csv")
    x_data, y_data = temp.create_data(None)
    print(temp.calculate_rho_saturation(x_data, y_data, rho_value_to_test))


def load_mult_detector_group_and_cluster():
    # Loading in all the data form A11Benchmark into detector classes
    manager = DetectorLayer(1)
    for i in range(1, 68):
        manager.add_detector("low_level_detector",
                             "data/A1Benchmark/real_"+str(i)+".csv")
    g = LayerGenerator(manager)
    print(g.group_and_cluster())


def load_mult_detector_group_create_layer():
    # Loading in all the data form A11Benchmark into detector classes
    manager = DetectorLayer(1)
    for i in range(1, 68):
        manager.add_detector("low_level_detector",
                             "data/A1Benchmark/real_"+str(i)+".csv")
    g = LayerGenerator(manager)
    g.create_new_layer(1)


def tester_iso_forest():
    hManager = HierarchalManager()
    files = []

    path = r'pci-slowdown-data'
    extension = '.csv'

    for root, dirs_list, files_list in os.walk(path):
        for file_name in files_list:
            if os.path.splitext(file_name)[-1] == extension:
                file_name_path = os.path.join(root, file_name)
                files.append("pci-slowdown-data/" + file_name)
    hManager.create_base_layer(files=files)
    for i in range(0, 7):
        hManager.mLayers[0].mDetector_map[1].graph_data(
            method="iso", contam=0.005*(i+1))


def data_length():
    hManager = HierarchalManager()
    files = []

    path = r'pci-slowdown-data'
    extension = '.csv'

    for root, dirs_list, files_list in os.walk(path):
        for file_name in files_list:
            if os.path.splitext(file_name)[-1] == extension:
                file_name_path = os.path.join(root, file_name)
                files.append("pci-slowdown-data/" + file_name)
    hManager.create_base_layer(files=files)
    for i in range(0, len(hManager.mLayers[0].mDetector_map)):
        print(len(hManager.mLayers[0].mDetector_map[i].mHistory))


def run_graph():
    hManager = HierarchalManager()
    files = []

    path = r'pci-slowdown-data'
    extension = '.csv'

    for root, dirs_list, files_list in os.walk(path):
        for file_name in files_list:
            if os.path.splitext(file_name)[-1] == extension:
                file_name_path = os.path.join(root, file_name)
                files.append("pci-slowdown-data/" + file_name)
    hManager.create_base_layer(files=files)
    for i in range(0, len(hManager.mLayers[0].mDetector_map)):
        hManager.mLayers[0].mDetector_map[i].graph_data()


def tester_std1():
    hManager = HierarchalManager()
    files = []

    path = r'pci-slowdown-data'
    extension = '.csv'

    for root, dirs_list, files_list in os.walk(path):
        for file_name in files_list:
            if os.path.splitext(file_name)[-1] == extension:
                file_name_path = os.path.join(root, file_name)
                files.append("pci-slowdown-data/" + file_name)
    hManager.create_base_layer(files=files)
    for i in range(0, 5):
        hManager.mLayers[0].mDetector_map[1].graph_data(
            method="ts", contam=0.005*(i+1))


def tester_std(rho):
    hManager = HierarchalManager()
    files = []
    path = r'pci-slowdown-data'
    extension = '.csv'

    for root, dirs_list, files_list in os.walk(path):
        for file_name in files_list:
            if os.path.splitext(file_name)[-1] == extension:
                file_name_path = os.path.join(root, file_name)
                files.append("pci-slowdown-data/" + file_name)
    hManager.create_base_layer(files=files)
    hManager.mLayers[0].mDetector_map[1].graph_data(
        method="ts", rho=rho)


def tune_rho(detector_id):
    hManager = HierarchalManager()
    files = []

    path = r'pci-slowdown-data'
    extension = '.csv'

    for root, dirs_list, files_list in os.walk(path):
        for file_name in files_list:
            if os.path.splitext(file_name)[-1] == extension:
                file_name_path = os.path.join(root, file_name)
                files.append("pci-slowdown-data/" + file_name)
    hManager.create_base_layer(files=files)
    # hManager.mLayers[0].mDetector_map[1].graph_data(method="ts")
    detector = hManager.mLayers[0].mDetector_map[detector_id]
    x_data, y_data = detector.create_data(None)
    rho_value_to_test = 3.010027210823807e-10
    it = 1
    while True:
        saturation = detector.calculate_rho_saturation(
            x_data, y_data, rho_value_to_test)
        if saturation > 0.0001:
            print("Iteration: " + str(it))
            print("Rho: " + str(rho_value_to_test))
            print("Saturation: " + str(saturation*100) + "%")
            print("-----------------")
            rho_value_to_test = rho_value_to_test - rho_value_to_test * 0.001
            return
        else:
            print("Iteration: " + str(it))
            print("Rho: " + str(rho_value_to_test))
            print("Saturation: " + str(saturation*100) + "%")
            print("-----------------")
            break
        it += 1
    # load_mult_detector_graph()

# load_single_detector_graph()

# load_single_detector_graph_anomaly([1300, 1400])

# load_single_detector_find_anomalies()

# rho_estimation(1.5)


# load_mult_detector_group_and_cluster()
# load_mult_detector_group_create_layer()
# data_length()

# run_graph()  # Contam would be different if not all send data at same point
# tester_iso_forest()
# tester_std(rho=3.010027210823807e-8)
# tune_rho(1)
def tester_std(rho):
    hManager = HierarchalManager()
    files = []
    path = r'pci-slowdown-data'
    extension = '.csv'

    for root, dirs_list, files_list in os.walk(path):
        for file_name in files_list:
            if os.path.splitext(file_name)[-1] == extension:
                file_name_path = os.path.join(root, file_name)
                files.append("pci-slowdown-data/" + file_name)
    hManager.create_base_layer(files=files)
    hManager.mLayers[0].mDetector_map[1].graph_data(
        method="ts", rho=rho)


def test_new_time_series(detector_id):
    hManager = HierarchalManager()
    files = []

    path = r'pci-slowdown-data'
    extension = '.csv'

    for root, dirs_list, files_list in os.walk(path):
        for file_name in files_list:
            if os.path.splitext(file_name)[-1] == extension:
                file_name_path = os.path.join(root, file_name)
                files.append("pci-slowdown-data/" + file_name)
    hManager.create_base_layer(files=files)
    hManager.mLayers[0].mDetector_map[detector_id].graph_data_with_anomaly(x_range=[
        240, 360])
    # Check left edge to see if it is meaningful


def chunk_events(detector_id):
    hManager = HierarchalManager()
    files = []

    path = r'pci-slowdown-data'
    extension = '.csv'

    for root, dirs_list, files_list in os.walk(path):
        for file_name in files_list:
            if os.path.splitext(file_name)[-1] == extension:
                file_name_path = os.path.join(root, file_name)
                files.append("pci-slowdown-data/" + file_name)
    hManager.create_base_layer(files=files)
    hManager.mLayers[0].mDetector_map[detector_id].find_unique_events()


chunk_events(8)

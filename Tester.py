from Detector import Detector
from DetectorLayer import DetectorLayer
from LayerGenerator import LayerGenerator
from HierarchalManager import HierarchalManager
from runner import Runner
import os
import threading
import matplotlib.pyplot as plt
import numpy as np


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


def chunk_events(detector_id, interval_index=0, target=1):
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
    hManager.mLayers[0].mDetector_map[detector_id].find_unique_events(
        index=interval_index, target_ev=target)


def stream_events(detector_id):
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
    target = 1
    for i in range(1, 16):
        interval_index = i * 50
        hManager.mLayers[0].mDetector_map[detector_id].find_unique_events(
            right_index=interval_index, width=min(interval_index, 400), target_ev=target)


def threaded_function(detector, interval_index, target, result, lock):
    ans = detector.find_unique_events(
        right_index=interval_index, width=min(interval_index, 400), target_ev=target)
    print("[threaded_function]: Waiting for lock.")
    print(ans)
    with lock:
        result.update({detector.mID: ans})
        print("[threaded_function]: Releasing lock.")


def multithreaded_stream(num_detectors, start_interval, total_intervals):
    """
    Pull the data from the database that we run the ALGO on
    """
    hManager = HierarchalManager()
    runner = Runner()
    detectors = runner.pull()
    hManager.create_base_layer(detectors=detectors)
    TARGET = 1
    """
    For every non-overlapping chunk, we want to create a thread that finds the 
    """
    for iteration in range(start_interval, total_intervals):
        result_queue = dict()
        STARTING_INTERVAL_INDEX = iteration * 50
        lock = threading.Lock()
        threads = []
        for i in range(num_detectors):
            current_detector = hManager.mLayers[0].mDetector_map[str(i)]
            thread = threading.Thread(
                target=threaded_function, args=(current_detector, STARTING_INTERVAL_INDEX, TARGET, result_queue, lock))
            threads.append(thread)
        for thread in threads:
            print("Starting Thread")
            thread.start()
        for thread in threads:
            print("Ending Thread")
            thread.join()
        """
        Count how often every anomaly shows up
        """
        queue_counter = dict()
        for anomalies in result_queue.values():
            for val in anomalies:
                if val in queue_counter.keys():
                    queue_counter[val] = queue_counter[val] + 1
                else:
                    queue_counter[val] = 1
        print("[multithreaded_stream]:")
        print(queue_counter)
        """
        For every detector, we want to plot it and the interval it just searched
        """
        TIME_DIFFERENCE_PER_POINT = 600000
        for i in range(num_detectors):
            current_detector = hManager.mLayers[0].mDetector_map[str(i)]
            interval_index = 50*(iteration - 2)
            x_data, y_data = current_detector.get_used_data(
                right_index=interval_index, width=min(interval_index, 400))
            plt.subplot(4, 3, 12)
            plt.plot(x_data, y_data)
            anomalies_found = []
            if current_detector.mID in result_queue:
                anomalies_found = result_queue[current_detector.mID]
            """
            If there were any anomalies in this detector, verify that it was found in another detector
            """
            if len(anomalies_found) > 0:
                for anomaly in anomalies_found:
                    count = queue_counter[anomaly]
                    for i in range(1, 5):  # Parameter, give small window of points
                        if (float(anomaly) - TIME_DIFFERENCE_PER_POINT*i) in queue_counter:
                            count = 2
                            count += queue_counter[(float(anomaly) -
                                                    TIME_DIFFERENCE_PER_POINT*i)]
                    for i in range(1, 5):
                        if (float(anomaly) + TIME_DIFFERENCE_PER_POINT*i) in queue_counter:
                            count = 2
                            count += queue_counter[(float(anomaly) +
                                                    TIME_DIFFERENCE_PER_POINT*i)]
                    if count > 1:
                        plt.plot(anomaly, np.mean(y_data), marker="x", markersize=5,
                                 markeredgecolor="red", markerfacecolor="green")

        plt.show()
        print("[multithreaded-stream]: ")
        print(result_queue)


def stream_events_ll():
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
    target = 1
    for detector in hManager.mLayers[0].mDetector_map.values():
        detector.graph()


stream_events_ll()
# multithreaded_stream(num_detectors=56, start_interval=0, total_intervals=16)

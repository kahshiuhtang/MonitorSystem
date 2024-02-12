from HierarchalManager import HierarchalManager
import matplotlib.pyplot as plt
import requests
from DBRunner import DBRunner
import datetime
import json
import threading
import numpy as np


class Streamer:
    def __init__(self, database="kahshiuh", host="localhost", user="kahshiuh", password="kahshiuh", port="5432"):
        self.db_runner = DBRunner(
            database=database, host=host, user=user, password=password, port=port)
        self.detectors = None
        self.alert_api_url = "http://localhost:9093/api/v2/alerts"

    def run(self):
        # self.db_runner.pull_from_prometheus()
        self.detectors = self.db_runner.pull_from_db()
        self.db_runner.shutdown()

        self.multithreaded_stream(len(self.detectors), 0, 16)

    def send_alerts(self):
        curr_time = datetime.datetime.utcnow()
        start_time = curr_time.isoformat("T") + "Z"

        current_time = datetime.datetime.utcnow()
        end_time = current_time + datetime.timedelta(seconds=30)
        end_time = end_time.isoformat("T") + "Z"
        data = [{'status': 'firing',
                'receiver': 'Anomaly',
                 'labels': {
                     'alertname': 'Anomaly Detected',
                     'severity': 'Warning'
                 },
                 'annotations': {
                     'summary': 'Alert summary here'
                 },
                 'startsAt': str(start_time),
                 'endsAt': str(end_time)
                 }]
        headers = {'Accept': 'application/json',
                   'Content-Type': 'application/json; charset=utf-8', }
        requests.post(url=self.alert_api_url,
                      headers=headers, data=json.dumps(data, ensure_ascii=True))
        return False

    def stream_events(self, detector_id):
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

    def threaded_function(self, detector, interval_index, target, result, lock):
        ans = detector.find_unique_events(
            right_index=interval_index, width=min(interval_index, 400), target_ev=target)
        print("[threaded_function]: Waiting for lock.")
        with lock:
            result.update({detector.mID: ans})
            print("[threaded_function]: Releasing lock.")

    def multithreaded_stream(self, num_detectors, start_interval, total_intervals):
        """
        Pull the data from the database that we run the ALGO on
        """
        hManager = HierarchalManager()
        detectors = self.detectors
        hManager.create_base_layer(detectors=detectors)
        TARGET = 1
        """
        For every non-overlapping chunk, we want to create a thread that finds the 
        """
        for iteration in range(start_interval, total_intervals):
            all_possible_anomalies_found = dict()
            STARTING_INTERVAL_INDEX = iteration * 50
            APAF_lock = threading.Lock()  # refers to dict two lines above
            threads = []
            for i in range(num_detectors):
                current_detector = hManager.mLayers[0].mDetector_map[str(i)]
                thread = threading.Thread(
                    target=self.threaded_function, args=(current_detector, STARTING_INTERVAL_INDEX, TARGET, all_possible_anomalies_found, APAF_lock))
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
            for anomaly in all_possible_anomalies_found.values():
                for val in anomaly:
                    if val in queue_counter.keys():
                        queue_counter[val] = queue_counter[val] + 1
                    else:
                        queue_counter[val] = 1
            """
            For every detector, we want to plot it and the interval it just searched
            """
            TIME_DIFFERENCE_PER_POINT = 600000
            for i in range(num_detectors):
                current_detector = hManager.mLayers[0].mDetector_map[str(i)]
                interval_index = 50*(iteration - 2)
                x_data, y_data = current_detector.get_used_data(
                    right_index=interval_index, width=min(interval_index, 400))
                anomalies_found = []
                if current_detector.mID in all_possible_anomalies_found:
                    anomalies_found = all_possible_anomalies_found[current_detector.mID]
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
            print("[multithreaded-stream]: Possible Anomalies Found...")
            print(all_possible_anomalies_found)
            plt.show()


s = Streamer()
# s.send_alerts()

s.run()

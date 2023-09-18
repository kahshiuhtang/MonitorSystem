import csv
from Rule import Rule
import matplotlib.pyplot as plt
import numpy as np
import math


class Detector:
    def __init__(self, id, detector_type, detector_level, lower_level_detectors):
        self.mName = ""
        self.mID = id
        self.mHistory = dict()
        self.mColumns = []
        self.mColumns_mapping = dict()
        self.mRules = []
        self.mDetectorLevel = detector_level
        self.mDetector_type = detector_type
        self.mX_data = None
        self.mY_data = None
        self.anomaly_data = None
        self.mLowerLevelDetectorIDs = lower_level_detectors

    #
    def squared_hellinger_distance(self, lambda_p, lambda_q, rho):
        return 1 - math.e ** (-rho * ((math.sqrt(lambda_p) - math.sqrt(lambda_q)) ** 2) / 2)

    # Rho should be set so the detector score saturates at 1 for a small percent of time
    # Paper says 0.1%
    def calculate_rho_saturation(self, x_data, y_data, rho):
        saturation = 0
        for idx in range(1, len(x_data)):
            left_arr = y_data[0:idx]
            right_arr = y_data[idx:]
            lambda_p = np.mean(left_arr)
            lambda_q = np.mean(right_arr)
            dissimilarity = self.squared_hellinger_distance(
                lambda_p, lambda_q, rho)
            if dissimilarity == 1:
                saturation += 1
        return float(saturation / len(x_data))

    def run_time_series(self, x_data, y_data, threshold):
        points = []
        maximum_value = -1
        max_timestamp = 0
        for idx in range(1, len(x_data)):
            left_arr = y_data[0:idx]
            right_arr = y_data[idx:]
            lambda_p = np.mean(left_arr)
            lambda_q = np.mean(right_arr)
            dissimilarity = self.squared_hellinger_distance(
                lambda_p, lambda_q, 0.00005)
            if threshold < dissimilarity:
                points.append((x_data[idx], y_data[idx]))
            if dissimilarity > maximum_value:
                maximum_value = dissimilarity
                max_timestamp = idx
        return points, maximum_value, max_timestamp

    def find_anomaly(self, threshold=0.25):
        x_data, y_data, _ = self.create_data(None)
        return self.run_time_series(x_data, y_data, threshold)

    # Some sort of rule system to check
    def add_rule(self, is_numeric, error_range):
        self.mRules.append(Rule(is_numeric, error_range))

    def check_rules(self, timestamp):
        if not isinstance(self.mHistory[timestamp], list):
            print("[Error]: No rules exist")
            return False
        for rule in self.mRules:
            if rule.is_valid(self.mHistory[timestamp][self.mColumns_mapping["value"]]) == False:
                return False
        return True

    # Used for simulating the data
    def next(self, timestamp):
        if len(self.mHistory) == 0:
            print("Error: No data found")
            return
        if str(timestamp) in self.mHistory:
            if self.mHistory[str(timestamp)][self.mColumns_mapping["is_anomaly"]] == "1":
                self.alert(timestamp)
            if self.check_rules(timestamp) == False:
                self.alert(timestamp)
            return self.mHistory[str(timestamp)]
        print("Timestamp not within range for detector " + str(self.mID))

    # Print out the detector ID and the anomaly
    def alert(self, timestamp):
        if len(self.mHistory) == 0:
            print("Error: No data found")
            return
        if str(timestamp) in self.mHistory:
            print("ALERT: Anomaly on detector " + str(self.mID))

    # Load in the all data, X:timestamp, Y:all values
    def load_from_file(self, file_path):
        with open(file_path, newline='') as csv_file:
            reader = csv.reader(csv_file, delimiter=',', quotechar='|')
            count = 0
            for row in reader:
                try:
                    if count != 0:
                        self.mHistory.update({row[0]: (row[1::])})
                    else:
                        self.mColumns = row[1::]
                        count += 1
                except ValueError:
                    print("Error")
        self.map_columns()
        return

    # Set up a mapping of the column names to the values
    def map_columns(self):
        for i in range(0, len(self.mColumns)):
            self.mColumns_mapping.update({self.mColumns[i]: i})

    # Retrieve all the data from a given timestamp
    def get_all_values_from_timestamp(self, timestamp):
        if str(timestamp) in self.mHistory:
            return self.mHistory[str(timestamp)]

    # Retrieve a field's value at a specific timestamp
    def get_value_from_timestamp(self, timestamp, field_name):
        if str(timestamp) in self.mHistory:
            if field_name in self.mColumns_mapping:
                return self.mHistory[str(timestamp)][self.mColumns_mapping[field_name]]
            print("Field name does not exist")
            return
        print("Timestamp not available")

    # Graph the values of a specific graph: also show the anomalies
    def graph_data(self, x_range=None):
        x_data, y_data, anomaly_data = self.create_data(x_range)
        plt.plot(x_data, y_data)
        for anomaly in anomaly_data:
            plt.plot(anomaly[0], anomaly[1], marker="x", markersize=5,
                     markeredgecolor="red", markerfacecolor="green")
        plt.xlabel('Timestamp')
        plt.ylabel('Value')
        plt.title('Values over time for detector:' + str(self.mID))
        plt.show()

    # Create the x,y and anomaly data for the .csv file
    def create_data(self, x_range=None):
        if self.mX_data is not None and self.mY_data is not None and self.anomaly_data is not None:
            return self.mX_data, self.mY_data, self.anomaly_data
        x_data = []
        y_data = []
        anomaly_data = []
        for timestamp in self.mHistory:
            if (x_range is not None and x_range[0] <= int(timestamp) and int(timestamp) < x_range[1]) or x_range is None:
                x_data.append(int(timestamp))
                if isinstance(self.mHistory[timestamp], list):
                    y_data.append(float(
                        self.mHistory[timestamp][self.mColumns_mapping["value"]]))
                else:
                    y_data.append(float(self.mHistory[timestamp]))
                if isinstance(self.mHistory[timestamp], list) and int(self.mHistory[timestamp][self.mColumns_mapping["is_anomaly"]]) == 1:
                    anomaly_data.append((int(timestamp), float(
                        self.mHistory[timestamp][self.mColumns_mapping["value"]])))
        self.mX_data = x_data
        self.mY_data = y_data
        self.anomaly_data = anomaly_data
        return x_data, y_data, anomaly_data

    def load_from_memory(self, timestamps, data):
        for i in range(0, len(data)):
            self.mHistory.update({timestamps[i]: (data[i])})
        return

    # Only graph the 0,1's of the anomaly values

    def graph_anomaly(self, x_range=None):
        x_data = []
        anomaly_data = []
        for timestamp in self.mHistory:
            if (x_range is not None and int(timestamp) >= x_range[0] and int(timestamp) <= x_range[1]) or x_range is None:
                x_data.append(timestamp)
                anomaly_data.append(
                    int(self.mHistory[timestamp][self.mColumns_mapping["is_anomaly"]]))
        plt.plot(x_data, anomaly_data)
        plt.xlabel('Timestamp')
        plt.ylabel('Anomaly')
        plt.title('Anomaly over time for detector:' + str(self.mID))
        plt.show()

    def save_history(self, file_location):
        with open(file_location, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Timestamp", "value"])
            rows = []
            for idx in self.mHistory:
                rows.append([idx, self.mHistory[idx]])
            writer.writerows(rows)
        print("Finished saving detector " +
              str(self.mID) + " history to file.")
        return True

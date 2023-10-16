import csv
from Rule import Rule
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.cluster import KMeans
import numpy as np
import math
from numpy import unique
from numpy import where
from sklearn.cluster import KMeans


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
        self.mLowerLevelDetectorIDs = lower_level_detectors

    #
    def squared_hellinger_distance(self, lambda_p, lambda_q, rho):
        return 1 - math.e ** (-rho * ((math.sqrt(lambda_p) - math.sqrt(lambda_q)) ** 2) / 2)

    def run_isolation_forest(self, x_data, y_data, method="i-forest", contamination=0.05):
        if method == "i-forest":
            x = np.array(x_data)
            y = np.array(y_data)
            data = np.column_stack((x, y))
            clf = IsolationForest(contamination=contamination, random_state=4)
            clf.fit(data)
            outlier_labels = clf.predict(data)
            mean = np.mean(y)
            std = np.std(y)

            anomaly_x_data = []
            anomaly_y_data = []
            z_score = []
            print("Anomaly Z-Scores with contamination of "+str(contamination)+":")
            for i in range(0, len(outlier_labels)):
                if outlier_labels[i] == -1:
                    anomaly_x_data.append(x_data[i])
                    anomaly_y_data.append(y_data[i])
                    z = (y_data[i] - mean) / std
                    print(z)
                    z_score.append(z)
            return anomaly_x_data, anomaly_y_data

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
                    if count != 0 and len(row) >= 2:
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
    def graph_data(self, x_range=None, method="ts", contam=0.05, num_clusters=2, rho=1):
        x_data, y_data = self.create_data(x_range)
        # plt.plot(x_data, y_data)
        maxX = None
        maxY = None
        if method == "iso":
            maxX, maxY = self.run_isolation_forest(
                x_data, y_data, contamination=contam)
        elif method == "std":
            maxX, maxY = self.find_anomaly_std(x_data, y_data, 10)
        elif method == "ts":
            maxX, maxY = self.run_time_series(
                x_data[240:360], y_data[240:360], rho=rho)
        plt.plot(x_data, y_data)
        for i in range(0, len(maxX)):
            plt.plot(maxX[i], maxY[i], marker="x", markersize=5,
                     markeredgecolor="red", markerfacecolor="green")
        plt.xlabel('Timestamp')
        plt.ylabel('Value')
        plt.title('Values over time for detector:' + str(self.mID))
        plt.show()

    # Create the x,y and anomaly data for the .csv file
    def create_data(self, x_range=None):
        if self.mX_data is not None and self.mY_data is not None:
            return self.mX_data, self.mY_data
        x_data = []
        y_data = []
        for timestamp in self.mHistory:
            if (x_range is not None and x_range[0] <= int(timestamp) and int(timestamp) < x_range[1]) or x_range is None:
                x_data.append(int(timestamp))
                if isinstance(self.mHistory[timestamp], list):
                    y_data.append(float(
                        self.mHistory[timestamp][0]))
                else:
                    y_data.append(float(self.mHistory[timestamp]))
        self.mX_data = x_data
        self.mY_data = y_data
        return x_data, y_data

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

    def find_valid_rho(self, x_data, y_data, starting_rho=10):
        rho_test = starting_rho
        if starting_rho is None:
            rho_test = 10
        saturation = self.calculate_rho_saturation(x_data, y_data, rho_test)
        prev_rho = rho_test
        if saturation < 0.0001:
            return -1
        while saturation > 0.0001:
            print(prev_rho)
            prev_rho = rho_test
            # [NOTE] What is a good learn percentage
            rho_test = rho_test - rho_test * 0.1
            saturation = self.calculate_rho_saturation(
                x_data, y_data, rho_test)
        return prev_rho
    # Rho should be set so the detector score saturates at 1 for a small percent of time
    # Paper says 0.1%

    def calculate_rho_saturation(self, x_data, y_data, rho):
        saturation = 0
        actual_diss = []
        for idx in range(1, len(x_data) - 1):
            left_arr = y_data[0:idx]
            right_arr = y_data[idx:]
            lambda_p = np.mean(left_arr)
            lambda_q = np.mean(right_arr)
            dissimilarity = self.squared_hellinger_distance(
                lambda_p, lambda_q, rho)
            actual_diss.append(dissimilarity)
            if dissimilarity > 0.9:
                saturation += 1
        return float(saturation / len(x_data))

    def calculate_num_above_threshold(self, x_data, y_data, rho, threshold):
        above_threshold = 0
        for idx in range(1, len(x_data) - 1):
            left_arr = y_data[0:idx]
            right_arr = y_data[idx:]
            lambda_p = np.mean(left_arr)
            lambda_q = np.mean(right_arr)
            dissimilarity = self.squared_hellinger_distance(
                lambda_p, lambda_q, rho)
            if dissimilarity > threshold:
                above_threshold += 1
        return float(above_threshold / len(x_data))

    def find_valid_threshold(self, x_data, y_data, rho, threshold_start, desired_threshold=0.0001):
        thresh_test = threshold_start
        if thresh_test is None:
            thresh_test = 0.9
        above_thresh = self.calculate_num_above_threshold(
            x_data, y_data, rho, thresh_test)
        while above_thresh < desired_threshold:
            thresh_test = thresh_test - 0.001 * thresh_test
            above_thresh = self.calculate_num_above_threshold(
                x_data, y_data, rho, thresh_test)
        return thresh_test

    def find_anomaly_via_ts(self, x_data, y_data, rho_start=10, threshold_start=None):
        default_rho = 10
        rho = self.find_valid_rho(x_data, y_data, rho_start)
        threshold = 0.9
        if rho < 0:
            rho = default_rho
            threshold = self.find_valid_threshold(
                x_data, y_data, rho=default_rho, threshold_start=0.9)
        x_points = []
        y_points = []
        # diss = []
        for idx in range(1, len(x_data) - 1):
            left_arr = y_data[0:idx]
            right_arr = y_data[idx:]
            lambda_p = np.mean(left_arr)
            lambda_q = np.mean(right_arr)
            dissimilarity = self.squared_hellinger_distance(
                lambda_p, lambda_q, rho)
            # diss.append(dissimilarity)
            if dissimilarity > threshold:
                x_points.append(x_data[idx])
                y_points.append(y_data[idx])
        # print(diss)
        return x_points, y_points

    def find_events_via_ts(self, x_data, y_data, rho_start=10, desired_threshold=0.02):
        default_rho = 10
        rho = self.find_valid_rho(x_data, y_data, rho_start)
        threshold = 0.9
        if rho < 0:
            rho = default_rho
            threshold = self.find_valid_threshold(
                x_data, y_data, rho=default_rho, threshold_start=0.9, desired_threshold=desired_threshold)
        points = []
        point_diss = dict()
        for idx in range(1, len(x_data) - 1):
            left_arr = y_data[0:idx]
            right_arr = y_data[idx:]
            lambda_p = np.mean(left_arr)
            lambda_q = np.mean(right_arr)
            dissimilarity = self.squared_hellinger_distance(
                lambda_p, lambda_q, rho)
            point_diss.update({idx: dissimilarity})
        ordered_diss = sorted(point_diss.items(),
                              key=lambda x: x[1], reverse=True)
        ordered_diss = ordered_diss[:int(0.4*len(x_data))]
        for tup in ordered_diss:
            points.append([x_data[tup[0]], y_data[tup[0]]])
        kmeans = KMeans(n_clusters=3, random_state=0,
                        n_init="auto").fit(points)
        return kmeans.cluster_centers_

    def graph_events(self):
        x_data, y_data = self.create_data()
        event_points = self.find_events_via_ts(
            x_data, y_data)
        plt.plot(x_data, y_data)
        for i in range(0, len(event_points)):
            plt.plot(event_points[i][0], event_points[i][1], marker="x", markersize=5,
                     markeredgecolor="red", markerfacecolor="green")
        plt.xlabel('Timestamp')
        plt.ylabel('Value')
        plt.title('Values over time for detector:' + str(self.mID))
        plt.show()
        return

    def graph_data_with_anomaly(self, x_range=None, rho_start=None, threshold_start=None):
        x_data, y_data = self.create_data()
        x_FULL = x_data
        y_FULL = y_data
        maxX = None
        maxY = None
        if x_range is not None:
            x_data = x_data[x_range[0]:x_range[1]]
            y_data = y_data[x_range[0]:x_range[1]]
            plt.axvline(x=x_FULL[x_range[0]], color='y',
                        label='axvline - full height')
            plt.axvline(x=x_FULL[x_range[1]], color='y',
                        label='axvline - full height')
        maxX, maxY = self.find_anomaly_via_ts(
            x_data, y_data, rho_start=rho_start, threshold_start=threshold_start)
        plt.plot(x_FULL, y_FULL)
        for i in range(0, len(maxX)):
            plt.plot(maxX[i], maxY[i], marker="x", markersize=5,
                     markeredgecolor="red", markerfacecolor="green")
        plt.xlabel('Timestamp')
        plt.ylabel('Value')
        plt.title('Values over time for detector:' + str(self.mID))
        plt.show()
        return

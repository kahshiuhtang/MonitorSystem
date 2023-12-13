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
        self.mAnomalyCollection = dict()

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

    def graph(self):
        x_data, y_data = self.create_data()
        if (len(x_data)) == 0:
            return
        plt.plot(x_data, y_data)
        plt.xlabel('Timestamp')
        plt.ylabel('Value')
        plt.title('Values over time for detector:' +
                  str(self.mID) + "," + str(self.mName))
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
            self.mHistory.update({timestamps[i]: [data[i]]})
        return

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

    def squared_hellinger_distance(self, lambda_p, lambda_q, rho):
        return 1 - math.e ** (-rho * ((math.sqrt(lambda_p) - math.sqrt(lambda_q)) ** 2) / 2)

    def find_valid_rho(self, x_data, y_data, starting_rho=10):
        # print("Finding valid rho")
        rho_test = starting_rho
        if starting_rho is None:
            rho_test = 10
        saturation = self.calculate_rho_saturation(x_data, y_data, rho_test)
        prev_rho = rho_test
        if saturation < 0.0001:
            # print("Invalid Rho Saturation")
            return -1
        max_iteration = 200
        current_iteration = 0
        # print("Starting Rho Finding")
        while saturation > 0.0001 and current_iteration < max_iteration:
            # print("Rho finding iteration")
            current_iteration = current_iteration + 1
            prev_rho = rho_test
            # [NOTE] What is a good learn percentage
            rho_test = rho_test - rho_test * 0.1
            saturation = self.calculate_rho_saturation(
                x_data, y_data, rho_test)
        return prev_rho
    # Rho should be set so the detector score saturates at 1 for a small percent of time
    # Paper says 0.1%

    def calculate_rho_saturation(self, x_data, y_data, rho):
        # print("Calculating Saturation")
        saturation = 0
        actual_diss = []
        left_sum = np.sum(y_data[0:1])
        right_sum = np.sum(y_data[1:])
        left_size = 1
        right_size = len(y_data) - 1
        for idx in range(1, len(x_data) - 1):
            val = y_data[idx]
            left_sum += val
            right_sum -= val
            left_size += 1
            right_size -= 1
            lambda_p = left_sum / left_size
            lambda_q = right_sum / right_size
            dissimilarity = self.squared_hellinger_distance(
                lambda_p, lambda_q, rho)
            actual_diss.append(dissimilarity)
            if dissimilarity > 0.9:
                saturation += 1
        # print("Finished Calculating Saturation")
        return float(saturation / len(x_data))

    def calculate_num_above_threshold(self, x_data, y_data, rho, threshold):
        above_threshold = 0
        left_sum = np.sum(y_data[0:1])
        right_sum = np.sum(y_data[1:])
        left_size = 1
        right_size = len(y_data) - 1
        for idx in range(1, len(x_data) - 1):
            val = y_data[idx]
            left_sum += val
            right_sum -= val
            left_size += 1
            right_size -= 1
            lambda_p = left_sum / left_size
            lambda_q = right_sum / right_size
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
        flip_directions = 1
        iterations = 0
        while above_thresh < desired_threshold and iterations < 500:
            iterations = iterations + 1
            if above_thresh < 0.0000001:
                flip_directions = -1
            thresh_test = thresh_test - 0.001 * thresh_test * flip_directions
            above_thresh = self.calculate_num_above_threshold(
                x_data, y_data, rho, thresh_test)
        return thresh_test

    def find_events_via_ts(self, x_data, y_data, y_data_mean, rho_start=10, desired_threshold=0.02):
        # print("Finding event via Time Series")
        default_rho = 10
        interval_mean = np.mean(y_data)
        rho = self.find_valid_rho(x_data, y_data, rho_start)
        threshold = 0.9
        y_data_max = np.max(y_data)
        y_data_min = np.min(y_data)
        x_data_max = np.max(x_data)
        x_data_min = np.min(x_data)  # Get caught in middle third is the goal
        if rho < 0:
            rho = default_rho
            threshold = self.find_valid_threshold(
                x_data, y_data, rho=default_rho, threshold_start=0.9, desired_threshold=desired_threshold)
        points = []
        point_diss = dict()
        avg_diss = []
        left_sum = np.sum(y_data[0:1])
        right_sum = np.sum(y_data[1:])
        left_size = 1
        right_size = len(y_data) - 1
        for idx in range(1, len(x_data) - 1):
            val = y_data[idx]
            left_sum += val
            right_sum -= val
            left_size += 1
            right_size -= 1
            lambda_p = left_sum / left_size
            lambda_q = right_sum / right_size
            dissimilarity = self.squared_hellinger_distance(
                lambda_p, lambda_q, rho)
            avg_diss.append(dissimilarity)
            point_diss.update({idx: dissimilarity})
        ordered_diss = sorted(point_diss.items(),
                              key=lambda x: x[1], reverse=True)
        ordered_diss = ordered_diss[:int(0.035*len(x_data))]
        uniq_points = []
        for tup in ordered_diss:
            points.append(
                [x_data[tup[0]], y_data[tup[0]], interval_mean, y_data_mean, y_data_max, y_data_min, x_data_max, x_data_min])
            uniq_points.append(tup[1])
        return points

    def graph_events_by_chunks(self, chunks=36):
        x_data, y_data = self.create_data()
        chunk_size = int(len(x_data) / chunks)
        y_mean = np.mean(y_data)
        for i in range(0, chunks):
            event_points = []
            event_points = self.find_events_via_ts(
                x_data[i*chunk_size:(i+1)*chunk_size], y_data[i*chunk_size:(i+1)*chunk_size], y_mean)
            for j in range(0, len(event_points)):
                plt.plot(event_points[j][0], event_points[j][1], marker="x", markersize=5,
                         markeredgecolor="red", markerfacecolor="green")
        plt.plot(x_data, y_data)
        plt.xlabel('Timestamp')
        plt.ylabel('Value')
        plt.title('Values over time for detector:' + str(self.mID))
        plt.show()
        return
    # find_unique_events
    #   find_events_via_ts
    #       find_valid_rho
    #       find_valid_threshold
    #           calculate_num_above_threshold
    #           calculate_rho_saturation
    #           squared_hellinger_distance
    #   filter_possible_event_points

    def find_unique_events(self, chunks=8, right_index=0, width=0, target_ev=1):
        x_data, y_data = self.get_data_interval(right_index, width)
        if len(x_data) == 0 or len(y_data) == 0:
            return []
        print("[find_unique_events]: Acquired Data")
        # Used to chunk each interval
        WHOLE_CHUNK_INTERVAL = int(len(x_data) / chunks)
        # Used to increase the interval
        small_chunk_interval = int(len(x_data) / (chunks*2-1))
        y_mean = np.mean(y_data)
        y_std = np.std(y_data)
        y_max = np.max(y_data)
        y_min = np.min(y_data)
        all_possible_events = []
        target_events = target_ev
        current_events = 0
        event_points = []
        best_events_score = 1000000
        lowest_events_score = 1000000
        best_events_points = []
        max_iterations = 10
        current_iteration = 0
        print("Setting up constants")
        print("Looping through intervals")
        while target_events != current_events and current_iteration < max_iterations:
            # print("Iteration: " + str(current_iteration))
            current_iteration = current_iteration + 1
            WHOLE_CHUNK_INTERVAL = int(len(x_data) / chunks)
            # Used to increase the interval
            small_chunk_interval = int(len(x_data) / (chunks*2-1))
            # print("Trying to find events in chunks")
            for i in range(0, chunks*2 - 1):
                # print("Looking in chunk " + str(i))
                event_points = self.find_events_via_ts(
                    x_data[i*small_chunk_interval:(i)*small_chunk_interval+WHOLE_CHUNK_INTERVAL], y_data[i*small_chunk_interval:(i)*small_chunk_interval+whole_chunk_interval], y_mean)
                for item in event_points:
                    all_possible_events.append(item)
            compartment_intervals = int(len(x_data) / 4) - 1
            chunk_int = int(len(x_data) / compartment_intervals)
            range_arr = dict()
            for i in range(0, compartment_intervals):
                range_arr[i] = [x_data[(i)*chunk_int], x_data[(i+1)*chunk_int]]

            event_points, ids = self.filter_possible_event_points(
                all_possible_events, y_max, y_min, range_arr)
            current_events = ids
            lowest_events_score = min(lowest_events_score, current_events)
            if abs(target_events - current_events) < abs(target_events - best_events_score):
                best_events_points = event_points
                best_events_score = current_events
            if current_events > target_events:
                chunks = chunks + 1
            elif current_events < target_events:
                chunks = chunks - 1
            if chunks == 0:
                break
        if abs(target_events - current_events) > abs(target_events - best_events_score):
            event_points = best_events_points
        if target_events < best_events_score and len(event_points) > 2:
            event_points = []
        self.add_points_to_collection(event_points)
        x_min = np.min(x_data)
        x_max = np.max(x_data)
        keys = []
        for key in self.mAnomalyCollection.keys():
            if key > x_min and key < x_max and self.mAnomalyCollection[key] > 1:
                keys.append(key)
        return keys
    '''
    Important points should occur in more than one interval -> Create boxes and try to see if more than one point is in box
    Important points will be in an interval that has great change
    '''

    def filter_possible_event_points(self, poss_events, y_max, y_min, compartments):
        num_in_compartment = dict()
        item_id_to_compartment = dict()
        for idx, item in enumerate(poss_events):
            for key in compartments.keys():
                if item[0] >= compartments[key][0] and item[0] < compartments[key][1]:
                    if key in num_in_compartment.keys():
                        num_in_compartment[key] = num_in_compartment[key] + 1
                    else:
                        num_in_compartment[key] = 1
                    item_id_to_compartment[idx] = key
                    break
        ans = []
        ids = dict()
        range_buffer = int(len(compartments) * 0.15)
        for idx, item in enumerate(poss_events):
            x_coord = item[0]
            interval_y_max = item[4]
            interval_y_min = item[5]
            if interval_y_max - interval_y_min > 0.5*y_max:
                if len(item_id_to_compartment) == 0 and idx == 0:
                    continue
                if idx not in item_id_to_compartment.keys():
                    continue
                compartment_id = item_id_to_compartment[idx]
                if num_in_compartment[compartment_id] > 1:
                    valid = True
                    for id in ids.keys():
                        if compartment_id < id + range_buffer and compartment_id > id - range_buffer:
                            valid = False
                            ids[id].append(x_coord)
                            break
                    if valid == True:
                        ans.append([x_coord, 0])
                        ids.update(
                            {compartment_id: [x_coord]})
        new_ans = []
        for id in ids.values():
            mean = np.mean(np.array(id))
            new_ans.append([mean, 0])
        return new_ans, len(ids)

    def add_points_to_collection(self, points):
        for point in points:
            flagged = False
            for x_val in self.mAnomalyCollection.keys():
                if abs(x_val - point[0]) < 120000 * 10:
                    flagged = True
                    self.mAnomalyCollection[x_val
                                            ] = self.mAnomalyCollection[x_val] + 1
            if not flagged:
                self.mAnomalyCollection[point[0]] = 1
        print("Anomaly Collection")
        print(self.mAnomalyCollection)

    def get_data_interval(self, index, width):
        x_data, y_data = self.create_data()
        if index == 0:
            return x_data, y_data
        if index > len(x_data):
            print("Error: dataset too small")
        if width == 0:
            x_data[:index], y_data[:index]
        if index - width < 0:
            print("Error: dataset too small")
        return x_data[index-width: index], y_data[index-width: index]

    def get_used_data(self, right_index=0, width=0):
        x_data, y_data = self.get_data_interval(right_index, width)
        return x_data, y_data

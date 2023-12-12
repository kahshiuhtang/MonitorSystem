import requests
import json
import psycopg2

from Detector import Detector
from DetectorLayer import DetectorLayer
from LayerGenerator import LayerGenerator
from HierarchalManager import HierarchalManager
import os
"""
Time Per Epoch: epoch_time{instance=~"${ml_job}"}
System CPU Usage: (1 - avg(rate(node_cpu_seconds_total{mode="idle", instance=~"${node}"}[1m])))*100
System Memory Usage: node_memory_Buffers_bytes{instance=~"${node}", job="node-exporter"}+node_memory_Cached_bytes{instance=~"${node}", job="node-exporter"}

GPU Temp: DCGM_FI_DEV_GPU_TEMP{instance=~"${instance}", gpu=~"${gpu}", job="dcgm"}
GPU Power Usage: DCGM_FI_DEV_POWER_USAGE{instance=~"${instance}", gpu=~"${gpu}", job="dcgm"}
GPU Utilization: DCGM_FI_DEV_GPU_UTIL{instance=~"${instance}", gpu=~"${gpu}", job="dcgm"}
GPU Framebuffer Mem Used: DCGM_FI_DEV_FB_USED{instance=~"${instance}", job="dcgm"}

PCIE TX: DCGM_FI_PROF_PCIE_TX_BYTES{instance=~"${instance}", job="dcgm"}
PCIE RX: DCGM_FI_PROF_PCIE_RX_BYTES{instance=~"${instance}", job="dcgm"}
GR Engine Utilization: DCGM_FI_PROF_GR_ENGINE_ACTIVE{instance=~"${instance}", gpu=~"${gpu}", job="dcgm"}
SM_Utilization: DCGM_FI_PROF_SM_OCCUPANCY{instance=~"${instance}", gpu=~"${gpu}", job="dcgm"}
GPU SM Clocks: DCGM_FI_DEV_SM_CLOCK{instance=~"${instance}", gpu=~"${gpu}", job="dcgm"}*1000000

{"name": "", "query": "", "time": ""}
"""
conn = psycopg2.connect(database="kahshiuh",
                        host="localhost",
                        user="kahshiuh",
                        password="kahshiuh",
                        port="5432")
cursor = conn.cursor()


def pull_metrics(query_url, query, time, isGPU):
    query_url = f'{query_url}/api/v1/query?query={query}{time}'
    print(query_url)
    res = requests.get(query_url)
    res_dict = res.json()
    if res_dict["status"] == "success":
        data = []
        res = "result"
        if "results" in res_dict["data"]:
            res = "results"
        if len(res_dict["data"][res]) == 0:
            print("No Results")
            return None
        for metric_point in res_dict["data"][res]:
            curr_data_point = []
            instance = ""
            gpu = ""
            if isGPU == "true":
                if "gpu" in metric_point["metric"]:
                    gpu = metric_point["metric"]["gpu"]

            if "instance" in metric_point["metric"]:
                instance = metric_point["metric"]["instance"]

            if "value" in metric_point:
                valueArray = metric_point["value"]
                if isGPU == "true":
                    curr_data_point.append(
                        (valueArray[0], instance, gpu, float(valueArray[1])))
                else:
                    curr_data_point.append(
                        (valueArray[0], instance, float(valueArray[1])))
            elif "values" in metric_point:
                valueArray = metric_point["values"]
                for data_point in valueArray:
                    if isGPU == "true":
                        curr_data_point.append(
                            (data_point[0], instance, gpu, float(data_point[1])))
                    else:
                        curr_data_point.append(
                            (data_point[0], instance, float(data_point[1])))
            else:
                return None
            data.append(curr_data_point)
        return data
    print("Failed Request")
    return None


def store_metrics(cursor, conn, table, value_names, value_types, data):
    if data == None:
        return
    for arr in data:
        cursor.executemany("""INSERT INTO {} {} VALUES {};""".format(table, value_names, value_types),
                           arr)
    conn.commit()


def find_all_unique(cursor, conn, table, column_name):
    if len(column_name) == 2:
        cursor.execute(
            """SELECT DISTINCT {}, {} FROM  {} WHERE realtime_col >= NOW() - INTERVAL '12 hours' ORDER BY {}, {};""".format(column_name[0], column_name[1], table, column_name[0], column_name[1]))
        results = cursor.fetchall()
        for row in results:
            print(len(row))
        print()
    else:
        cursor.execute(
            """SELECT DISTINCT {} FROM  {} WHERE realtime_col >= NOW() - INTERVAL '12 hours';""".format(column_name[0], table))
        results = cursor.fetchall()
        for row in results:
            print(len(row))
        print()
    return results


def load_into_detector(data, dID, dLevel):
    if data is None or len(data) == 0:
        return
    detector = Detector(dID, "", dLevel, [])
    timestamps = []
    data1 = []
    tup_len = len(data[0])
    for i in range(len(data)):
        timestamps.append(data[i][0])
        data1.append(data[i][tup_len - 1])
    detector.load_from_memory(timestamps, data1)
    detector.graph()
    return detector


def load_from_database(cursor, conn, table, constraints):
    if (constraints[0] == ""):
        return
    if len(constraints) == 2:
        cursor.execute(
            """SELECT timestamp_col, value_col FROM {} WHERE realtime_col >= NOW() - INTERVAL '12 hours' AND gpu = '{}' AND instance = '{}';""".format(table, constraints[0], constraints[1]))
    else:
        cursor.execute(
            """SELECT timestamp_col, value_col FROM {} WHERE realtime_col >= NOW() - INTERVAL '12 hours' AND instance = '{}';""".format(table, str(constraints[0])))
    results = cursor.fetchall()
    for row in results:
        print(row)
    return results


# {"name": "Time Per Epoch", "query": "epoch_time", "time": "[48h]", "db_name": "time_per_epoch", "gpu_metric": "false"}
# {"name": "SM_Utilization","query": 'DCGM_FI_PROF_SM_OCCUPANCY', "time": "[5h]",  "db_name": "sm_utilization", "gpu_metric": "true"}
queries = [
    {"name": "System CPU Usage",
     "query": 'avg(rate(node_cpu_seconds_total{mode="idle"}[1m]))', "time": "", "db_name": "system_cpu_usage", "gpu_metric": "false"},
    {"name": "System Memory Usage Buffer",
     "query": 'node_memory_Buffers_bytes{job="node-exporter"}', "time": "[1m]", "db_name": "memory_usage_buffer", "gpu_metric": "false"},
    {"name": "System Memory Usage Cached",
     "query": 'node_memory_Cached_bytes{job="node-exporter"}', "time": "[1m]", "db_name": "memory_usage_cached", "gpu_metric": "false"},
    {"name": "GPU Temp",
     "query": 'DCGM_FI_DEV_GPU_TEMP', "time": "[1m]",  "db_name": "gpu_temp", "gpu_metric": "true"},
    {"name": "GPU Power Usage",
     "query": 'DCGM_FI_DEV_POWER_USAGE', "time": "[1m]", "db_name": "gpu_power_usage", "gpu_metric": "true"},
    {"name": "GPU Utilization",
     "query": 'DCGM_FI_DEV_GPU_UTIL', "time": "[1m]", "db_name": "gpu_utilization", "gpu_metric": "true"},
    {"name": "GPU Framebuffer Mem Used",
     "query": ' DCGM_FI_DEV_FB_USED', "time": "[1m]", "db_name": "gpu_frame_buffer", "gpu_metric": "false"},
    {"name": "PCIE TX",
     "query": 'DCGM_FI_PROF_PCIE_TX_BYTES', "time": "[1m]",  "db_name": "pcie_tx", "gpu_metric": "false"},
    {"name": "PCIE RX",
     "query": 'DCGM_FI_PROF_PCIE_RX_BYTES', "time": "[1m]",  "db_name": "pcie_rx", "gpu_metric": "false"},
    {"name": "GR Engine Utilization",
     "query": 'DCGM_FI_PROF_GR_ENGINE_ACTIVE', "time": "[1m]", "db_name": "gr_engine_utilization", "gpu_metric": "true"},
    {"name": "GPU SM Clocks",
     "query": 'DCGM_FI_DEV_SM_CLOCK', "time": "[1m]", "db_name": "gpu_sm_clocks", "gpu_metric": "true"},
]


def run(cursor, conn, queries):
    for query in queries:
        print(query)
        data = pull_metrics('http://localhost:9090',
                            query["query"], query["time"], query["gpu_metric"])
        value_names = "(timestamp_col, instance, value_col)"
        value_types = "(%s, %s, %s)"
        if query["gpu_metric"] == "true":
            value_names = "(timestamp_col, instance, gpu, value_col)"
            value_types = "(%s, %s, %s, %s)"
        store_metrics(cursor, conn, query["db_name"],
                      value_names, value_types, data)


def pull(cursor, conn, queries):
    for query in queries:
        if query["gpu_metric"] == "true":
            unique_pairs = find_all_unique(cursor, conn, query["db_name"], [
                "gpu", "instance"])
            for pair in unique_pairs:
                data = load_from_database(cursor, conn, query["db_name"], pair)
        else:
            unique_pairs = find_all_unique(cursor, conn, query["db_name"], [
                "instance"])
            for pair in unique_pairs:
                data = load_from_database(cursor, conn, query["db_name"], pair)


def TEST_pull_metrics():
    for query in queries:
        print(json.dumps(pull_metrics('http://localhost:9090',
                                      query["query"], query["time"], queries["gpu_metric"]), indent=1))


def TEST_pull_metric(idx):
    print(json.dumps(pull_metrics('http://localhost:9090',
                                  queries[idx]["query"], queries[idx]["time"], queries[idx]["gpu_metric"]), indent=1))


def TEST_store_metric(idx, cursor, conn):
    data = pull_metrics('http://localhost:9090',
                        queries[idx]["query"], queries[idx]["time"], queries[idx]["gpu_metric"])
    value_names = "(timestamp_col, instance, value_col)"
    value_types = "(%s, %s, %s)"
    if queries[idx]["gpu_metric"] == "true":
        value_names = "(timestamp_col, instance, gpu, value_col)"
        value_types = "(%s, %s, %s, %s)"
    load_into_detector(data, 1, 1)
    """
        store_metrics(cursor, conn, queries[idx]["db_name"],
                  value_names, value_types, data)
    """


def TEST_load_from_database(cursor, conn, table, time_start, time_end):
    print(load_from_database(cursor, conn, table, time_start, time_end))


# Doesn't work for 0 (no epoch time)
# DCGM_FI_PROF_SM_OCCUPANCY, 11 doesnt work -> maybe because DCGM
# TEST_pull_metric(11)
pull(cursor, conn, queries)
# print(json.dumps(pull_metrics('http://localhost:9090',
#   'node_memory_Buffers_bytes{job="node-exporter"}', '[15m]'), indent=2))
# TEST_store_metric(8, cursor, conn)
# TEST_load_from_database(cursor, conn, "gpu_temp", 1701874891.33, 1701875101.33)
conn.close()

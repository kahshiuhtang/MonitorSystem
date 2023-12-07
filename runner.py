import requests
import json
import psycopg2

from Detector import Detector
from DetectorLayer import DetectorLayer
from LayerGenerator import LayerGenerator
from HierarchalManager import HierarchalManager
import os

conn = psycopg2.connect(database="kahshiuh",
                        host="localhost",
                        user="kahshiuh",
                        password="kahshiuh",
                        port="5432")
cursor = conn.cursor()
# cursor.execute("""INSERT INTO time_per_epoch (timestamp_col, value_col) VALUES (%s, %s);""",
#              (15, 10))
# conn.commit()


def pull_metrics(query_url, query, time, isGPU):
    query_url = f'{query_url}/api/v1/query?query={query}{time}'
    res = requests.get(query_url)
    res_dict = res.json()
    if res_dict["status"] == "success":
        data = []
        res = "result"
        if "results" in res_dict["data"]:
            res = "results"
        if len(res_dict["data"][res]) == 0:
            print("No Results")
            return
        if "value" in res_dict["data"][res][0]:
            instance = res_dict["data"][res][0]["metric"]["instance"]
            if isGPU == "true":
                value = res_dict["data"][res][0]["value"]
                gpu = res_dict["data"][res][0]["metric"]["gpu"]
                for data_point in value:
                    data.append(
                        (data_point[0], instance, gpu, float(data_point[1])))
            else:
                value = res_dict["data"][res][0]["value"]
                for data_point in value:
                    data.append(
                        (data_point[0], instance, float(data_point[1])))
        elif "values" in res_dict["data"][res][0]:
            value = res_dict["data"][res][0]["values"]
            instance = res_dict["data"][res][0]["metric"]["instance"]
            if isGPU == "true":
                gpu = res_dict["data"][res][0]["metric"]["gpu"]
                for data_point in value:
                    data.append(
                        (data_point[0], instance, gpu, float(data_point[1])))
            else:
                for data_point in value:
                    data.append(
                        (data_point[0], instance, float(data_point[1])))
        else:
            return None
        return data
    print("Failed Request")
    return


def store_metrics(cursor, conn, table, value_names, value_types, data):
    cursor.executemany("""INSERT INTO {} {} VALUES {};""".format(table, value_names, value_types),
                       data)
    conn.commit()


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


def load_from_database(cursor, conn, table, time_start, time_end):
    cursor.execute(
        """SELECT * FROM  {} WHERE timestamp_col > %s AND timestamp_col < %s;""".format(table), (time_start, time_end))
    results = cursor.fetchall()
    for row in results:
        print(row)
    return results


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
queries = [{"name": "Time Per Epoch", "query": "epoch_time", "time": "[24h]", "db_name": "time_per_epoch", "gpu_metric": "false"},
           {"name": "System CPU Usage",
               "query": 'avg(rate(node_cpu_seconds_total{mode="idle"}[1m]))', "time": "", "db_name": "system_cpu_usage", "gpu_metric": "false"},
           {"name": "System Memory Usage Buffer",
               "query": 'node_memory_Buffers_bytes{job="node-exporter"}', "time": "[1m]", "db_name": "memory_usage_buffer", "gpu_metric": "false"},
           {"name": "System Memory Usage Cached",
               "query": 'node_memory_Cached_bytes{job="node-exporter"}', "time": "[1m]", "db_name": "memory_usage_cached", "gpu_metric": "true"},
           {"name": "GPU Temp",
               "query": 'DCGM_FI_DEV_GPU_TEMP{job="dcgm"}', "time": "[3h]", "db_name": "time_per_epoch", "db_name": "gpu_temp", "gpu_metric": "true"},
           {"name": "GPU Power Usage",
               "query": 'DCGM_FI_DEV_POWER_USAGE{job="dcgm"}', "time": "[3h]", "db_name": "time_per_epoch", "db_name": "gpu_power_usage", "gpu_metric": "true"},
           {"name": "GPU Utilization",
               "query": 'DCGM_FI_DEV_GPU_UTIL{ job="dcgm"}', "time": "[12h]", "db_name": "time_per_epoch", "db_name": "gpu_utilization", "gpu_metric": "true"},
           {"name": "GPU Framebuffer Mem Used",
               "query": ' DCGM_FI_DEV_FB_USED{job="dcgm"}', "time": "[1m]", "db_name": "time_per_epoch", "db_name": "gpu_frame_buffer", "gpu_metric": "false"},
           {"name": "PCIE TX",
               "query": 'DCGM_FI_PROF_PCIE_TX_BYTES{job="dcgm"}', "time": "[12h]", "db_name": "time_per_epoch", "db_name": "pcie_tx", "gpu_metric": "false"},
           {"name": "PCIE RX",
               "query": 'DCGM_FI_PROF_PCIE_RX_BYTES{ job="dcgm"}', "time": "[3h]", "db_name": "time_per_epoch", "db_name": "pcie_rx", "gpu_metric": "false"},
           {"name": "GR Engine Utilization",
               "query": 'DCGM_FI_PROF_GR_ENGINE_ACTIVE{job="dcgm"}', "time": "[3h]", "db_name": "time_per_epoch", "db_name": "gr_engine_utilization", "gpu_metric": "true"},
           {"name": "SM_Utilization",
               "query": 'DCGM_FI_PROF_SM_OCCUPANCY{job="dcgm"}', "time": "[12h]", "db_name": "time_per_epoch", "db_name": "sm_utilization", "gpu_metric": "true"},
           {"name": "GPU SM Clocks",
               "query": 'DCGM_FI_DEV_SM_CLOCK{job="dcgm"}', "time": "[3h]", "db_name": "time_per_epoch", "db_name": "gpu_sm_clocks", "gpu_metric": "true"},
           ]


def TEST_pull_metrics():
    for query in queries:
        print(json.dumps(pull_metrics('http://localhost:9090',
                                      query["query"], query["time"]), indent=1))


def TEST_pull_metric(idx):
    print(json.dumps(pull_metrics('http://localhost:9090',
                                  queries[idx]["query"], queries[idx]["time"]), indent=1))


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


# TEST_pull_metrics()
# print(json.dumps(pull_metrics('http://localhost:9090',
#   'node_memory_Buffers_bytes{job="node-exporter"}', '[15m]'), indent=2))
TEST_store_metric(8, cursor, conn)
# TEST_load_from_database(cursor, conn, "gpu_temp", 1701874891.33, 1701875101.33)
conn.close()

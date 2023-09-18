from Detector import Detector
from DetectorLayer import DetectorLayer
from LayerGenerator import LayerGenerator


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
    x_data, y_data, _ = temp.create_data(None)
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


# load_mult_detector_graph()

# load_single_detector_graph()

# load_single_detector_graph_anomaly([1300, 1400])

# load_single_detector_find_anomalies()

# rho_estimation(1.5)

# load_mult_detector_group_and_cluster()
load_mult_detector_group_create_layer()

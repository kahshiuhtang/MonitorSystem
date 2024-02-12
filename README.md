# NVidia GPU Cluster Monitor

## About

This is a automatic cluster monitor system. Prometheus runs on a GPU cluster and monitors and stores the metrics. Data is pulled from Prometheus every 15 minutes and stored into a PostgreSQL database. The data is then pulled by the program, and runs the anomaly detection algorithm on it. If any possible anomalies are detected, a POST request is sent to a Docker container running Prometheus AlertManager, which will send an alert to either email or Slack.

Two sample datasets are available, pci-slowdown-data and test-pci. 

## Running

To install all requirements, run:

```bash

$ pip install -r requirements.txt

```

Running Prometheus on your cluster can be ported locally so this program can collect the data that has been scraped. Be sure the match the port numbers from your local computer to address used to scrape the API endpoint.

You can run this with Docker, likely with a command similar to the following:

```bash
$ docker run -p 9090:9090 -v /path/to/prometheus.yml:/etc/prometheus/prometheus. yml prom/prometheus
```

This specifies you are running Prometheus locally on port 9090 and specifies the path for the yml file.


To setup PostgreSQL, you will need to setup a database with tables to store the metrics collected from Prometheus. This can be configured in the Streamer.py file or in the DBRunner.py file.

![Example Database](images/Postgres.jpg.jpg)


To run the AlertManager in a Docker container, run:

```bash
$ docker run -p 9093:9093 -d prom/alertmanager
```

This will run AlertManager locally on port 9093


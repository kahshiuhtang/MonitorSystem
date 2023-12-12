import requests
import json

query = 'http://localhost:9090/api/v1/query?query=DCGM_FI_PROF_SM_OCCUPANCY[1d]'
res = requests.get(query)
data = res.json()
print(data)
for i in data["data"]["result"]:
    print(i["metric"]["gpu"])

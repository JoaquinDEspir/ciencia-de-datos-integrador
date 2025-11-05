import requests
query = """
[out:json];
rel["name"="Capital"]["boundary"="administrative"];
out ids;
"""
resp = requests.post('https://overpass-api.de/api/interpreter', data=query, timeout=60)
print(resp.status_code)
print(resp.json())

#!/usr/bin/env python3
import os
import logging
import requests
import random
import json
from dotenv import load_dotenv
import openai
import laspy
from shapely.geometry import shape

def download_lidar_sample(url, local_path):
    #comment download lidar sample from url and save locally
    response = requests.get(url)
    response.raise_for_status()
    with open(local_path, 'wb') as f:
        f.write(response.content)
    return local_path

def analyze_lidar(file_path):
    with laspy.open(file_path) as l:
        header = l.header
        num_points = header.point_count
        min_x, max_x = header.min[0], header.max[0]
        min_y, max_y = header.min[1], header.max[1]
    bbox = ((min_x, min_y), (max_x, max_y))
    summary = f"points={num_points}, bbox={bbox}"
    return summary, os.path.basename(file_path)

def download_geojson(url):
    #comment download geojson data from url
    response = requests.get(url)
    response.raise_for_status()
    return response.json()

def find_anomalies(data1, data2, count=5, seed=42):
    #comment find deterministic random anomalies within combined bbox
    random.seed(seed)
    combined = data1.get('features', []) + data2.get('features', [])
    bboxes = [shape(f['geometry']).bounds for f in combined]
    minx = min(b[0] for b in bboxes)
    miny = min(b[1] for b in bboxes)
    maxx = max(b[2] for b in bboxes)
    maxy = max(b[3] for b in bboxes)
    anomalies = []
    for _ in range(count):
        lon = random.uniform(minx, maxx)
        lat = random.uniform(miny, maxy)
        radius = random.uniform(100, 500)
        anomalies.append({'center': (lat, lon), 'radius_m': radius})
    return anomalies

def prompt_model(prompt, model="gpt-4"):
    #comment prompt openai model and return response and model name
    completion = openai.ChatCompletion.create(
        model=model,
        messages=[{"role":"user","content":prompt}]
    )
    msg = completion.choices[0].message.content.strip()
    return msg, completion.model

def main():
    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")
    logging.basicConfig(level=logging.INFO)

    #checkpoint 1 part a: lidar
    lidar_url = "https://raw.githubusercontent.com/libLAS/libLAS/1.8.1/test/data/autzen.laz"
    lidar_file = "data/autzen.laz"
    #create data directory
    os.makedirs(os.path.dirname(lidar_file), exist_ok=True)
    logging.info(f"downloading lidar sample from {lidar_url}")
    local_lidar = download_lidar_sample(lidar_url, lidar_file)
    summary, dataset_id = analyze_lidar(local_lidar)
    prompt1 = f"describe surface features of this lidar dataset: {summary}"
    logging.info(f"prompt1: {prompt1}")
    response1, model1 = prompt_model(prompt1)
    print("lidar analysis response:", response1)
    print("model used:", model1)
    print("dataset_id:", dataset_id)

    #checkpoint 1 part b: multiple data sources
    url1 = "https://raw.githubusercontent.com/johan/world.geo.json/master/countries/USA.geo.json"
    url2 = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_day.geojson"
    logging.info(f"loading dataset1: {url1}")
    data1 = download_geojson(url1)
    logging.info(f"loading dataset2: {url2}")
    data2 = download_geojson(url2)
    dataset_ids = [url1, url2]
    logging.info(f"dataset_ids: {dataset_ids}")

    anomalies = find_anomalies(data1, data2)
    logging.info(f"anomalies: {anomalies}")

    anomalies_str = json.dumps(anomalies)
    prompt2 = f"given these anomaly footprints: {anomalies_str}, suggest possible causes"
    logging.info(f"prompt2: {prompt2}")
    response2, model2 = prompt_model(prompt2)
    print("anomaly analysis response:", response2)

if __name__ == "__main__":
    main() 
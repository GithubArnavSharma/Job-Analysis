"""
Enter flask run to run it 
Control C to escape
Use 3.8 (env)
"""

import numpy as np 
import pandas as pd
import pickle
from flask import Flask, request, jsonify, render_template
from math import cos, asin, sqrt, pi
from sentence_transformers import SentenceTransformer, util
import pymupdf
import re
import torch
import ast

model = SentenceTransformer('all-MiniLM-L6-v2')

app = Flask(__name__)

with open('loc2coord.pkl', 'rb') as f:
    loc2coord = pickle.load(f)

df = pd.read_csv("postings_updated.csv")

def distance(lat1, lon1, lat2, lon2):
    r = 6371 # km
    p = pi / 180

    a = 0.5 - cos((lat2-lat1)*p)/2 + cos(lat1*p) * cos(lat2*p) * (1-cos((lon2-lon1)*p))/2
    return (2 * r * asin(sqrt(a))) / 1.609


def file_to_text(file):
  text = ""
  doc = pymupdf.open(file)
  for page in doc:
    text += str(page.get_text())
  text = re.sub(r'[^a-zA-Z0-9 .]', '', text)
  return text

def text_to_similar(text, num_jobs, salary_min=None, city_provided="", dist_limit=None, entry_level=False):
  new_df = df.copy()
  if entry_level:
    new_df = new_df[new_df["formatted_experience_level"] == "Entry level"]

  if loc2coord.get(city_provided.lower()) is not None:
    lat, lng = loc2coord[city_provided.lower()]
    new_df = new_df[new_df["lat"].notna()]
    new_df["Distance"] = new_df.apply(lambda row: distance(row.lat, row.lng, lat, lng), axis=1)
    new_df = new_df[new_df["Distance"] <= dist_limit]

  if salary_min is not None:
    new_df = new_df[(new_df["med_salary"].notna()) & (new_df["med_salary"] >= salary_min)]

  all_encodings = [np.asarray(np.matrix(new_df["encoding_jobs"].iloc[i]))[0].astype(float) for i in range(new_df.shape[0])]
  #print(all_encodings[0])
  #encodings_tensor = torch.tensor(np.stack(all_encodings))
  sims = np.array(model.similarity(np.array(model.encode(text)).astype(float), all_encodings))[0]
  """
  all_encodings = [np.array(new_df["encoding_jobs"].iloc[i]) for i in range(new_df.shape[0])]
  sims = np.array(model.similarity(model.encode([text]), all_encodings))[0]
  """
  index2sim = {i: s for i, s in enumerate(sims)}
  index2sim = dict(sorted(index2sim.items(), key=lambda item: item[1], reverse=True))
  the_jobs_index = list(index2sim.keys())[:num_jobs]
  specific_sims = list(index2sim.values())[:num_jobs]

  BOLD = '<b>'  
  END = '</b>' 
  output = "<br>"
  for i, s in zip(the_jobs_index, specific_sims):
    row = new_df.iloc[i] 
    link = str(row["job_posting_url"])


    the_salary = row["med_salary"]
    if pd.isna(the_salary):
      the_salary = "N/A"
    output += "<h4>" + BOLD + str(row["title"]) + " | " + f"{s*100:.1f}% match" + " | " + \
          f"${the_salary} salary<br>" + "Location: " + str(row["location"]) + END + "<br>" \
          + f'<a href="{link}" target="_blank">Apply Here</a></h4><br><br>'
    output += str(new_df.iloc[i]["description"]).strip()+"<br><br><br>"


  return output 


@app.route('/jobs',methods=['POST'])
def jobs():
  try:
    file = request.files.get("resume") 
    num_jobs = int(request.form.get("entries"))
    salary_min = int(request.form.get("salary"))
    city_provided = str(request.form.get("city"))
    state_provided = str(request.form.get("state"))
    dist_limit = int(request.form.get("miles"))
    entry_level = (str(request.form.get("yes_or_no")) == "yes")
  except:
     return jsonify({'message': ""})

  city_provided = f"{city_provided}, {state_provided}"

  file = pymupdf.open(stream=file.read(), filetype="pdf")

  the_text = ""
  for page in file:
      the_text += page.get_text()
  #print(the_text) 

  output = text_to_similar(the_text, num_jobs, salary_min, city_provided, dist_limit, entry_level)
  return jsonify({'message': output})

@app.route("/")
def index():
   return render_template("front.html")

if __name__ == '__main__':
    app.run(debug=True)

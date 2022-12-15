#-H "accept: application/json" -H "Content-Type: application/json" -d '{"prompt": "Niko the kobold stalked carefully down the alley, his small scaly figure obscured by a dusky cloak that fluttered lightly in the cold winter breeze. Holding up his tail to keep it from dragging in the dirty snow that covered the cobblestone, he waited patiently for the butcher to turn his attention from his stall so that he could pilfer his next meal: a tender-looking chicken. He crouched just slightly as he neared the stall to ensure that no one was watching, not that anyone would be dumb enough to hassle a small kobold. What else was there for a lowly kobold to do in a city? All that Niko needed to know was where to find the chicken and then how to make off with it. A soft thud caused Niko to quickly lift his head. Standing behind the stall where the butcher had been cutting his chicken,", "temperature": 0.5, "top_p": 0.9}'}

import requests, tqdm, json, os
prompt = "Niko the kobold stalked carefully down the alley, his small scaly figure obscured by a dusky cloak that fluttered lightly in the cold winter breeze. Holding up his tail to keep it from dragging in the dirty snow that covered the cobblestone, he waited patiently for the butcher to turn his attention from his stall so that he could pilfer his next meal: a tender-looking chicken. He crouched just slightly as he neared the stall to ensure that no one was watching, not that anyone would be dumb enough to hassle a small kobold. What else was there for a lowly kobold to do in a city? All that Niko needed to know was where to find the chicken and then how to make off with it. A soft thud caused Niko to quickly lift his head. Standing behind the stall where the butcher had been cutting his chicken,"
url = "http://localhost:5000/api/v1/generate"

model = input("What model are we using?\n")
sequences = int(input("Number of gens in each batch\n"))
total_sequences = 10

if os.path.exists("survey_input.json"):
    with open("survey_input.json", "r") as f:
        results = json.load(f)
results = {model: {"temp": {}, "top_k": {}, "rep_pen": {}}}
for temp_tenths in tqdm.tqdm(range(1, 21)):
    temp = float(temp_tenths)/10.0
    output = []
    while len(output) < total_sequences:
        x = requests.post(url, json = {"prompt": prompt, "temperature": temp, "n": sequences}, headers = {"Content-Type": "application/json", "accept": "application/json"})
        output.extend([item['text'] for item in x.json()['results']])
    results[model]["temp"][temp] = output[:10]

for temp_tenths in tqdm.tqdm(range(50, 101, 5)):
    temp = float(temp_tenths)/100.0
    output = []
    while len(output) < total_sequences:
        x = requests.post(url, json = {"prompt": prompt, "top_k": temp, "n": sequences}, headers = {"Content-Type": "application/json", "accept": "application/json"})
        output.extend([item['text'] for item in x.json()['results']])
    results[model]["top_k"][temp] = output[:10]

for temp_tenths in tqdm.tqdm(range(100, 131, 5)):
    temp = float(temp_tenths)/100.0
    output = []
    while len(output) < total_sequences:
        x = requests.post(url, json = {"prompt": prompt, "rep_pen": temp, "n": sequences}, headers = {"Content-Type": "application/json", "accept": "application/json"})
        output.extend([item['text'] for item in x.json()['results']])
    results[model]["rep_pen"][temp] = output[:10]

with open("survey_input.json", "w") as f:
    f.write(json.dumps(results, indent="\t"))
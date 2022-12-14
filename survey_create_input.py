#-H "accept: application/json" -H "Content-Type: application/json" -d '{"prompt": "Niko the kobold stalked carefully down the alley, his small scaly figure obscured by a dusky cloak that fluttered lightly in the cold winter breeze. Holding up his tail to keep it from dragging in the dirty snow that covered the cobblestone, he waited patiently for the butcher to turn his attention from his stall so that he could pilfer his next meal: a tender-looking chicken. He crouched just slightly as he neared the stall to ensure that no one was watching, not that anyone would be dumb enough to hassle a small kobold. What else was there for a lowly kobold to do in a city? All that Niko needed to know was where to find the chicken and then how to make off with it. A soft thud caused Niko to quickly lift his head. Standing behind the stall where the butcher had been cutting his chicken,", "temperature": 0.5, "top_p": 0.9}'}

import requests, tqdm, json
prompt = "Niko the kobold stalked carefully down the alley, his small scaly figure obscured by a dusky cloak that fluttered lightly in the cold winter breeze. Holding up his tail to keep it from dragging in the dirty snow that covered the cobblestone, he waited patiently for the butcher to turn his attention from his stall so that he could pilfer his next meal: a tender-looking chicken. He crouched just slightly as he neared the stall to ensure that no one was watching, not that anyone would be dumb enough to hassle a small kobold. What else was there for a lowly kobold to do in a city? All that Niko needed to know was where to find the chicken and then how to make off with it. A soft thud caused Niko to quickly lift his head. Standing behind the stall where the butcher had been cutting his chicken,"
url = "http://localhost:5000/api/v1/generate"

results = {"temp": {}, "top_k": {}, "rep_pen": {}}
for temp_tenths in tqdm.tqdm(range(1, 21)):
    temp = float(temp_tenths)/10.0
    x = requests.post(url, json = {"prompt": prompt, "temperature": temp}, headers = {"Content-Type": "application/json", "accept": "application/json"})
    results["temp"][temp] = [item['text'] for item in x.json()['results']]

for temp_tenths in tqdm.tqdm(range(50, 101, 5)):
    temp = float(temp_tenths)/100.0
    x = requests.post(url, json = {"prompt": prompt, "top_k": temp}, headers = {"Content-Type": "application/json", "accept": "application/json"})
    results["top_k"][temp] = [item['text'] for item in x.json()['results']]

for temp_tenths in tqdm.tqdm(range(100, 131, 5)):
    temp = float(temp_tenths)/100.0
    x = requests.post(url, json = {"prompt": prompt, "rep_pen": temp}, headers = {"Content-Type": "application/json", "accept": "application/json"})
    results["rep_pen"][temp] = [item['text'] for item in x.json()['results']]

with open("output.json", "w") as f:
    f.write(json.dumps(results, indent="\t"))
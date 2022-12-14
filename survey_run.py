import flask, json, flask_socketio, random, os, sys
app = flask.Flask(__name__, root_path=os.getcwd())
socketio = flask_socketio.SocketIO(app, async_method="eventlet", manage_session=False, cors_allowed_origins='*', max_http_buffer_size=10_000_000)

with open("survey_input.json", "r") as f:
    data = json.load(f)
    
if os.path.exists("survey_results.json"):
    with open("survey_results.json", "r") as f:
        results = json.load(f)
else:
    results = {"Which one feels more random": {}, 
               "Which one feels more creative": {},
               "Which one feels more repetative": {}
              }

@app.route('/')
def index():
    #choose temp, top_k, or rep_pen
    data_type = random.choice(['temp', 'temp', 'temp', 'temp', 'top_k', 'top_k', 'rep_pen'])
    a = random.choice([x for x in data[data_type]])
    b = random.choice([x for x in data[data_type]])
    while b == a:
        b = random.choice([x for x in data[data_type]])
        
    random_num=random.randint(-sys.maxsize, sys.maxsize)
    question = {"temp": "Which one feels more random", "top_k": "Which one feels more creative", "rep_pen": "Which one feels more repetative"}[data_type]
    if random_num not in results[question]:
        results[question][random_num] = {}
    results[question][random_num]['A'] = a
    results[question][random_num]['B'] = b
    
    a = random.choice(data[data_type][a])
    b = random.choice(data[data_type][b])
    results[question][random_num]['A value'] = a
    results[question][random_num]['B value'] = b
    
    return flask.render_template('survey.html', question=question, a=a, b=b, random_num=random_num)
    
    

@socketio.on('answer')
def answer(data):
    results[data['question']][int(data['id'])]['Answer'] = data['answer']
    with open("survey_results.json", "w") as f:
        f.write(json.dumps(results, indent="\t"))
    
    
socketio.run(app)
import pytest, time
import aiserver

#Test Model List:
test_models = [
                ('EleutherAI/gpt-neo-1.3B', {'key': False, 'gpu': False, 'layer_count': 24, 'breakmodel': True, 'url': False}), 
                ('gpt2', {'key': False, 'gpu': False, 'layer_count': 12, 'breakmodel': True, 'url': False}), 
                ('facebook/opt-350m', {'key': False, 'gpu': False, 'layer_count': 24, 'breakmodel': True, 'url': False})
              ]

@pytest.fixture
def client_data():
    app = aiserver.app
    #app.test_client_class = FlaskLoginClient
    client_conn = app.test_client()
    socketio_client = aiserver.socketio.test_client(app, flask_test_client=client_conn)
    #Clear out the connection message
    response = socketio_client.get_received()
    return (client_conn, app, socketio_client)


def get_model_menu(model):
    for menu in aiserver.model_menu:
        for item in aiserver.model_menu[menu]:
            if item[1] == model:
                for main_menu_line in aiserver.model_menu['mainmenu']:
                    if main_menu_line[1] == menu:
                        return (menu, main_menu_line, item)
    return None
    
def generate_story_data(client_data):
    (client, app, socketio_client) = client_data
    socketio_client.emit('message',{'cmd': 'submit', 'allowabort': False, 'actionmode': 0, 'chatname': None, 'data': ''})
    
    #wait until the game state turns back to start
    state = 'wait'
    new_text = None
    start_time = time.time()
    timeout = time.time() + 60*1
    while state == 'wait':
        if time.time() > timeout:
            break
        responses = socketio_client.get_received()
        for response in responses:
            response = response['args'][0]
            print(response)
            if response['cmd'] == 'setgamestate':
                state = response['data']
            elif response['cmd'] == 'updatechunk' or response['cmd'] == 'genseqs':
                new_text = response['data']
        time.sleep(0.1)
    
    assert new_text is not None

def test_basic_connection(client_data):
    (client, app, socketio_client) = client_data
    response = client.get("/")
    assert response.status_code == 200

def test_load_story_from_web_ui(client_data):
    (client, app, socketio_client) = client_data
    
    #List out the stories and make sure we have the sample story
    socketio_client.emit('message',{'cmd': 'loadlistrequest', 'data': ''})
    response = socketio_client.get_received()[0]['args'][0]['data']
    found_sample_story = False
    for story in response:
        if story['name'] == 'sample_story':
            found_sample_story = True
    assert found_sample_story
    
    #Click on the sample story, then click load
    socketio_client.emit('message',{'cmd': 'loadselect', 'data': 'sample_story'})
    socketio_client.emit('message',{'cmd': 'loadrequest', 'data': ''})
    
    #Wait until we get the data back from the load
    loaded_story = False
    timeout = time.time() + 60*2
    while not loaded_story:
        if time.time() > timeout:
            break
        responses = socketio_client.get_received()
        for response in responses:
            response = response['args'][0]
            if 'cmd' not in response:
                print(response)
                assert False
            if response['cmd'] == 'updatescreen':
                loaded_story = True
                story_text = response['data']
                break
    assert loaded_story
    
    #Verify that it's the right story data
    assert story_text == '<chunk n="0" id="n0" tabindex="-1">Niko the kobold stalked carefully down the alley, his small scaly figure obscured by a dusky cloak that fluttered lightly in the cold winter breeze. Holding up his tail to keep it from dragging in the dirty snow that covered the cobblestone, he waited patiently for the butcher to turn his attention from his stall so that he could pilfer his next meal: a tender-looking</chunk><chunk n="1" id="n1" tabindex="-1"> chicken. He crouched just slightly as he neared the stall to ensure that no one was watching, not that anyone would be dumb enough to hassle a small kobold. What else was there for a lowly kobold to</chunk><chunk n="2" id="n2" tabindex="-1"> do in a city? All that Niko needed to know was</chunk><chunk n="3" id="n3" tabindex="-1"> where to find the chicken and then how to make off with it.<br/><br/>A soft thud caused Niko to quickly lift his head. Standing behind the stall where the butcher had been cutting his chicken,</chunk>'  

@pytest.mark.parametrize("model, expected_load_options", test_models)
def test_load_model_from_web_ui(client_data, model, expected_load_options):
    (client, app, socketio_client) = client_data
    
    #Clear out any old messages
    response = socketio_client.get_received()
    
    (menu, menu_line, model_line) = get_model_menu(model)
    
    #Send the ai load model menu option
    socketio_client.emit('message',{'cmd': 'list_model', 'data': 'mainmenu'})
    response = socketio_client.get_received()[0]['args'][0]['data']
    assert menu_line in response
    
    #Send the click model menu option
    socketio_client.emit('message',{'cmd': 'list_model', 'data': menu, 'pretty_name': ""})
    response = socketio_client.get_received()[0]['args'][0]['data']
    assert model_line in response
    
    #Click the model
    socketio_client.emit('message',{'cmd': 'selectmodel', 'data': model})
    response = socketio_client.get_received()[0]['args'][0]
    #Check that we're getting the right load options
    print(response)
    assert response['key'] == expected_load_options['key']
    assert response['gpu'] == expected_load_options['gpu']
    assert response['layer_count'] == expected_load_options['layer_count']
    assert response['breakmodel'] == expected_load_options['breakmodel']
    assert response['url'] == expected_load_options['url']
    
    #Now send the load 
    socketio_client.emit('message',{'cmd': 'load_model', 'use_gpu': True, 'key': '', 'gpu_layers': str(expected_load_options['layer_count']), 'disk_layers': '0', 'url': '', 'online_model': ''})
    #wait until the game state turns back to start
    state = 'wait'
    start_time = time.time()
    timeout = time.time() + 60*2
    while state == 'wait':
        if time.time() > timeout:
            break
        responses = socketio_client.get_received()
        for response in responses:
            response = response['args'][0]
            if response['cmd'] == 'setgamestate':
                state = response['data']
        time.sleep(0.1)
    
    #Give it a second to get all of the settings, etc and clear out the messages
    responses = socketio_client.get_received()
    
    #check the model info to see if it's loaded
    socketio_client.emit('message',{'cmd': 'show_model', 'data': ''})
    response = socketio_client.get_received()[0]['args'][0]
    assert response == {'cmd': 'show_model_name', 'data': model}
    
    generate_story_data(client_data)
  
def test_load_GooseAI_from_web_ui(client_data):
    
    pytest.skip("unsupported configuration")

@pytest.mark.parametrize("model, expected_load_options", test_models)
def test_load_model_from_command_line(client_data, model, expected_load_options):
    (client, app, socketio_client) = client_data
    
    #Clear out any old messages
    response = socketio_client.get_received()
    
    (menu, menu_line, model_line) = get_model_menu(model)
    
    aiserver.general_startup("--model {}".format(model))
    
    aiserver.load_model(initial_load=True)
    
    #check the model info to see if it's loaded
    socketio_client.emit('message',{'cmd': 'show_model', 'data': ''})
    response = socketio_client.get_received()[0]['args'][0]
    assert response == {'cmd': 'show_model_name', 'data': model}
    
    generate_story_data(client_data)

def test_back_redo(client_data):
    (client, app, socketio_client) = client_data
    
    
    #Make sure we have known story in the ui
    test_load_story_from_web_ui(client_data)
    
    #Clear out any old messages
    response = socketio_client.get_received()
    
    #run a back action
    socketio_client.emit('message',{'cmd': 'back', 'data': ''})
    response = socketio_client.get_received()[0]['args'][0]
    assert response == {'cmd': 'removechunk', 'data': 3}
    
    #Run a redo action
    socketio_client.emit('message',{'cmd': 'redo', 'data': ''})
    response = socketio_client.get_received()[0]['args'][0]
    assert response == {'cmd': 'updatechunk', 'data': {'index': 3, 'html': '<chunk n="3" id="n3" tabindex="-1"> where to find the chicken and then how to make off with it.<br/><br/>A soft thud caused Niko to quickly lift his head. Standing behind the stall where the butcher had been cutting his chicken,</chunk>'}}
    
    #Go all the way back, then all the way forward
    socketio_client.emit('message',{'cmd': 'back', 'data': ''})
    response = socketio_client.get_received()[0]['args'][0]
    assert response == {'cmd': 'removechunk', 'data': 3}
    socketio_client.emit('message',{'cmd': 'back', 'data': ''})
    response = socketio_client.get_received()[0]['args'][0]
    assert response == {'cmd': 'removechunk', 'data': 2}
    socketio_client.emit('message',{'cmd': 'back', 'data': ''})
    response = socketio_client.get_received()[0]['args'][0]
    assert response == {'cmd': 'removechunk', 'data': 1}
    socketio_client.emit('message',{'cmd': 'back', 'data': ''})
    response = socketio_client.get_received()[0]['args'][0]
    assert response == {'cmd': 'errmsg', 'data': 'Cannot delete the prompt.'}
    socketio_client.emit('message',{'cmd': 'redo', 'data': ''})
    response = socketio_client.get_received()
    assert response == [{'name': 'from_server', 'args': [{'cmd': 'updatescreen', 'gamestarted': True, 'data': '<chunk n="0" id="n0" tabindex="-1">Niko the kobold stalked carefully down the alley, his small scaly figure obscured by a dusky cloak that fluttered lightly in the cold winter breeze. Holding up his tail to keep it from dragging in the dirty snow that covered the cobblestone, he waited patiently for the butcher to turn his attention from his stall so that he could pilfer his next meal: a tender-looking</chunk><chunk n="1" id="n1" tabindex="-1"> chicken. He crouched just slightly as he neared the stall to ensure that no one was watching, not that anyone would be dumb enough to hassle a small kobold. What else was there for a lowly kobold to</chunk>'}], 'namespace': '/'}, 
                        {'name': 'from_server', 'args': [{'cmd': 'texteffect', 'data': 1}], 'namespace': '/'}]
    socketio_client.emit('message',{'cmd': 'redo', 'data': ''})
    response = socketio_client.get_received()
    assert response == [{'name': 'from_server', 'args': [{'cmd': 'updatechunk', 'data': {'index': 2, 'html': '<chunk n="2" id="n2" tabindex="-1"> do in a city? All that Niko needed to know was</chunk>'}}], 'namespace': '/'}, 
                        {'name': 'from_server', 'args': [{'cmd': 'texteffect', 'data': 2}], 'namespace': '/'}]
    socketio_client.emit('message',{'cmd': 'redo', 'data': ''})
    response = socketio_client.get_received()
    assert response == [{'name': 'from_server', 'args': [{'cmd': 'updatechunk', 'data': {'index': 3, 'html': '<chunk n="3" id="n3" tabindex="-1"> where to find the chicken and then how to make off with it.<br/><br/>A soft thud caused Niko to quickly lift his head. Standing behind the stall where the butcher had been cutting his chicken,</chunk>'}}], 'namespace': '/'}, 
                        {'name': 'from_server', 'args': [{'cmd': 'texteffect', 'data': 3}], 'namespace': '/'}]
    
    
    

    
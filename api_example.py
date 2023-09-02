import requests

user = "User:"
bot = "Bot:"
ENDPOINT = "http://127.0.0.1:5000"
conversation_history = [] # using a list to update conversation history is more memory efficient than constantly updating a string

def get_prompt(user_msg):
    return {
        "prompt": f"{user_msg}",
        "use_story": False, #Needs to be set in KoboldAI webUI
        "use_memory": False, #Needs to be set in KoboldAI webUI
        "use_authors_note": False, #Needs to be set in KoboldAI webUI
        "use_world_info": False, #Needs to be set in KoboldAI webUI
        "max_context_length": 2048,
        "max_length": 120,
        "rep_pen": 1.0,
        "rep_pen_range": 2048,
        "rep_pen_slope": 0.7,
        "temperature": 0.7,
        "tfs": 0.97,
        "top_a": 0.8,
        "top_k": 0,
        "top_p": 0.5,
        "typical": 0.19,
        "sampler_order": [6,0,1,3,4,2,5], 
        "singleline": False,
        "sampler_seed": 69420, # Use specific seed for text generation?
        "sampler_full_determinism": False, # Always give same output with same settings?
        "frmttriminc": False, #Trim incomplete sentences
        "frmtrmblln": False, #Remove blank lines
        "stop_sequence": ["\n\n\n\n\n", f"{user}"]
        }

while True:
    try:
        user_message = input(f"{user}")

        if len(user_message.strip()) < 1:
            print(f"{bot}Please provide a valid input.")
            continue

        fullmsg = f"{conversation_history[-1] if conversation_history else ''}{user} {user_message}\n{bot} " # Add all of conversation history if it exists and add User and Bot names
        prompt = get_prompt(fullmsg) # Process prompt into KoboldAI API format
        response = requests.post(f"{ENDPOINT}/api/v1/generate", json=prompt) # Send prompt to API

        if response.status_code == 200:
            results = response.json()['results'] # Set results as JSON response
            text = results[0]['text'] # inside results, look in first group for section labeled 'text'
            response_text = text.split('\n')[0].replace("  ", " ") # Optional, keep only the text before a new line, and replace double spaces with normal ones
            conversation_history.append(f"{fullmsg}{response_text}\n") # Add the response to the end of your conversation history
        print(f"{bot} {response_text}")

    except Exception as e:
        print(f"An error occurred: {e}")

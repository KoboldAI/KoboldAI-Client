import requests

user = "User:"
bot = "Bot:"
ENDPOINT = "http://localhost:5000/api"
conversation_history = [] # using a list to update conversation history is more memory efficient than constantly updating a string

def get_prompt(user_msg):
    return {
        "prompt": f"{user_msg}",
        "use_story": "False", # Use the story from the KoboldAI UI, can be managed using other API calls (See /api for the documentation)
        "use_memory": "False", # Use the memnory from the KoboldAI UI, can be managed using other API calls (See /api for the documentation)
        "use_authors_note": "False", # Use the authors notes from the KoboldAI UI, can be managed using other API calls (See /api for the documentation)
        "use_world_info": "False", # Use the World Info from the KoboldAI UI, can be managed using other API calls (See /api for the documentation)
        "max_context_length": 2048, # How much of the prompt will we submit to the AI generator? (Prevents AI / memory overloading)
        "max_length": 100, # How long should the response be?
        "rep_pen": 1.1, # Prevent the AI from repeating itself
        "rep_pen_range": 2048, # The range to which to apply the previous
        "rep_pen_slope": 0.7, # This number determains the strength of the repetition penalty over time
        "temperature": 0.5, # How random should the AI be? In a low value we pick the most probable token, high values are a dice roll
        "tfs": 0.97, # Tail free sampling, https://www.trentonbricken.com/Tail-Free-Sampling/
        "top_a": 0.0, # Top A sampling , https://github.com/BlinkDL/RWKV-LM/tree/4cb363e5aa31978d801a47bc89d28e927ab6912e#the-top-a-sampling-method
        "top_k": 0, # Keep the X most probable tokens
        "top_p": 0.9, # Top P sampling / Nucleus Sampling, https://arxiv.org/pdf/1904.09751.pdf
        "typical": 1.0, # Typical Sampling, https://arxiv.org/pdf/2202.00666.pdf
        "sampler_order": [6,0,1,3,4,2,5], # Order to apply the samplers, our default in this script is already the optimal one. KoboldAI Lite contains an easy list of what the
        "stop_sequence": [f"{user}"], # When should the AI stop generating? In this example we stop when it tries to speak on behalf of the user.
        #"sampler_seed": 1337, # Use specific seed for text generation? This helps with consistency across tests.
        "singleline": "False", # Only return a response that fits on a single line, this can help with chatbots but also makes them less verbose
        "sampler_full_determinism": "False", # Always return the same result for the same query, best used with a static seed
        "frmttriminc": "True", # Trim incomplete sentences, prevents sentences that are unfinished but can interfere with coding and other non english sentences
        "frmtrmblln": "False", #Remove blank lines
        "quiet": "False" # Don't print what you are doing in the KoboldAI console, helps with user privacy
        }

while True:
    try:
        user_message = input(f"{user} ")

        if len(user_message.strip()) < 1:
            print(f"{bot}Please provide a valid input.")
            continue

        fullmsg = f"{conversation_history[-1] if conversation_history else ''}{user} {user_message}\n{bot}" # Add all of conversation history if it exists and add User and Bot names
        prompt = get_prompt(fullmsg) # Process prompt into KoboldAI API format
        response = requests.post(f"{ENDPOINT}/v1/generate", json=prompt) # Send prompt to API
        if response.status_code == 200:
            results = response.json()['results'] # Set results as JSON response
            text = results[0]['text'] # inside results, look in first group for section labeled 'text'
            response_text = text.split('\n')[0].replace("  ", " ") # Optional, keep only the text before a new line, and replace double spaces with normal ones
            conversation_history.append(f"{fullmsg}{response_text}\n") # Add the response to the end of your conversation history
        else:
            print(response)
        print(f"{bot} {response_text}")

    except Exception as e:
        print(f"An error occurred: {e}")
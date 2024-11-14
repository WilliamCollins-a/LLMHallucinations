

import re
import json
from transformers import LlamaForCausalLM, LlamaTokenizer

#Google Drive mount here
from google.colab import drive
drive.mount('/content/gdrive/')

#enabling import of custom module for colab
! cp /content/gdrive/MyDrive/Util.py . 

#set up file location here
import sys
sys.path.append('/content/gdrive/MyDrive')


# Load the LLaMA 2 model and tokenizer from Hugging Face
model_name = "meta-llama/Llama-2-7b-hf"  # Replace with your desired model name
tokenizer = LlamaTokenizer.from_pretrained(model_name)
model = LlamaForCausalLM.from_pretrained(model_name)

# Load the data from a JSON file
def load_json_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

# Generate initial response
def generate_response(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=50)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Feedback mechanism (simple rule: response must contain the keyword "refined")
def feedback(response):
    return "refined" in response.lower()

# Self-refine method
def self_refine(prompt, max_attempts):
    attempt = 0
    response = generate_response(prompt)
    
    while attempt < max_attempts:
        print(f"Attempt {attempt}: Generated Response = {response}")
        
        # Simple feedback mechanism
        if feedback(response):
            print(f"Response is acceptable!")
            break
        else:
            # Refine the response by appending a "refinement" request
            refined_prompt = f"{prompt}. Please refine the output."
            response = generate_response(refined_prompt)
            print(f"Refining Response to: {response}")
        
        attempt += 1

    return response

# Main function to load data and run self-refine
def main(json_file, max_attempts):
    data = load_json_data(json_file)
    
    for item in data:
        title = item['title']
        print(f"Refining response for: {title}")
        refined_response = self_refine(title, max_attempts)
        print(f"Final Refined Response: {refined_response}\n")

if __name__ == "__main__":
    # Example JSON data file path
    json_file = "data.json"
    max_attempts = 3
    main(json_file, max_attempts)
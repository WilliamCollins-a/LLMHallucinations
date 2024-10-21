## ZERO SHOT COT ##
"""run this in colab first"""
#!pip install transformers sentencepiece huggingface_hub
#!pip install bitsandbytes
#!pip install sentence-transformers
#!pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118
#from huggingface_hub import login
#login("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")

"""then this lot"""
#from google.colab import files
#uploaded = files.upload()  # Upload your JSON file
""" ^ this is used in colab in seprate cells to set up the envionrment, I've done a basic test on QA data thus far
you'll need to update your google drive and login."""


"""then this lot. line 48 restricts output to 200 tokens some of the response gets cut off so can be increased."""
# imports the required libarys
import json
from transformers import LlamaForCausalLM, LlamaTokenizer
from huggingface_hub import login

# Authenticate with Hugging Face 
login("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")  # hugging face token

# Load the LLaMA 2 chat model and tokenizer 
model_name = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = LlamaTokenizer.from_pretrained(model_name) # loads tokenizer that processes inputs for model
model = LlamaForCausalLM.from_pretrained(model_name) # loads the model

# Load the data from a JSON file
def load_json_data(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line.strip()))  # Load each line as a JSON object
    return data

# Define  JSON file path that contains QA data
json_file_path = r'/content/qa_data.json'
data = load_json_data(json_file_path) # loads Json file into the data variable

# Generate response with Chain of Thought (CoT) prompting
def generate_response_with_cot(knowledge, question):
    # Construct the prompt combines Knowledge and question
    cot_prompt = f"{knowledge} {question} Let's think step by step."
    inputs = tokenizer(cot_prompt, return_tensors="pt") # tokenizer and converts to tensors required
    outputs = model.generate(**inputs, max_new_tokens=200) # response restricted to 200 tokens in output, can increase current output cuts off!
    response = tokenizer.decode(outputs[0], skip_special_tokens=True) # decodes back to readable output
    return response

# Main function to load data and generate CoT responses
def main_with_cot(json_file):
    data = load_json_data(json_file)

    for item in data[:5]:  # Process only the first 5 items
        knowledge = item['knowledge'] # extract Knowledge from data set
        question = item['question'] # exctract question from data set

        print(f"Generating response for question: {question}")
        
        # Generate the response using CoT
        cot_response = generate_response_with_cot(knowledge, question)

        # Extract the final answer from the response
        final_answer = cot_response.split('Therefore,')[-1].strip().split('.')[0].strip()  # Get the part after "Therefore"

        # Print the reasoning steps and the final model answer
        print(f"CoT Response: {cot_response}\n")
        print(f"Model's Answer: {final_answer}\n")

if __name__ == "__main__":
    json_file = json_file_path  # Use the defined path variable
    main_with_cot(json_file)
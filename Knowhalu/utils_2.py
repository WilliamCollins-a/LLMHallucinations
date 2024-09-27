import pandas as pd
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import csv
#generates the datafram and indivual vectors/lists for each important aspect such as prompt and Info
def load_data_JSON(file,type_prompt):
    df = pd.read_json(file,lines = True)
    if type_prompt == 'general':
        prompt = df['user_query']
        correct = df['hallucination']
        gpt4 = df['chatgpt_response']
        Info = df['hallucination_spans'] 
    if type_prompt == 'qa':
        prompt = df['question']
        correct = df['right_answer']
        gpt4 = df['hallucinated_answer']
        Info= df['knowledge']
    if type_prompt == 'sum':
        prompt = df['document']
        correct = df['right_summary']
        gpt4 = df['hallucinated_summary']
        Info= []
    return df,prompt,correct,gpt4,Info

#instead of access token could just use hugginface hub login plus access token to remove all needs of repeat entry of access token for each run
def load_model(model_name,access):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, token=access)
    quantization_config = BitsAndBytesConfig(load_in_4bit=True)
    model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config,
    device_map="auto",
    token=access)
    return tokenizer,model
#model name "meta-llama/Llama-2-7b-chat-hf"

def write_out(filename,data):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)
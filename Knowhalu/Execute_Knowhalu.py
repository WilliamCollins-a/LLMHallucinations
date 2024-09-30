# import libraries
from Util import load_data_JSON, load_model, write_out
import pandas as pd
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
     

#load the data
general_df, prompt_g, correct_g, gtp4_g, Info_g = load_data_JSON('input/general_data.json', 'general')
qa_df, prompt_q, correct_q, gtp4_q, Info_q = load_data_JSON('input/qa.json', 'qa')
sum_df, prompt_s, correct_s, gtp4_s, Info_s = load_data_JSON('input/summarization_data.json', 'sum')
     

# Access input (API key, etc.)
access = input('Access code?')

#setup of model
tokenizer, model = load_model("meta-llama/Llama-2-7b-chat-hf", access)

#Knowhalu method
conda create --name knowhalu python=3.8
conda activate knowhalu
pip install -r requirements.txt



from utils_2.py import load_data_JSON, load_model
import pandas as pd
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    
import qa_relevance.py
import qa_query.py --model meta-llama/Llama-2-7b-chat-hf --form semantic --topk 2 --answer_type right --knowledge_type ground --query_selection None\n",
import qa_judge.py
    
import qa_judge.py

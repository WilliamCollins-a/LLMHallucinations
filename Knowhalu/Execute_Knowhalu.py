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
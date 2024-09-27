{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "conda create --name knowhalu python=3.8\n",
    "conda activate knowhalu\n",
    "pip install -r requirements.txt\n",
    "\n",
    "from utils_2.py import load_data_JSON, load_model\n",
    "import pandas as pd\n",
    "from huggingface_hub import login\n",
    "from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig\n",
    "\n",
    "import qa_relevance.py\n",
    "import qa_query.py --model meta-llama/Llama-2-7b-chat-hf --form semantic --topk 2 --answer_type right --knowledge_type ground --query_selection None\n",
    "import qa_judge.py\n",
    "\n",
    "import qa_judge.py\n"
   ]
  }
 ],
 "metadata": {
  "language_info": {
   "name": "python"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}

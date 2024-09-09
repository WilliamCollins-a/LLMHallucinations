import pandas as pd

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

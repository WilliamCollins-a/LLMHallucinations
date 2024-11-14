!pip install transformers sentencepiece huggingface_hub
!pip install bitsandbytes
!pip install sentence-transformers
!pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118
from huggingface_hub import login
login("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")

import json
import re
import matplotlib.pyplot as plt
from transformers import LlamaForCausalLM, LlamaTokenizer
from huggingface_hub import login


from google.colab import drive
drive.mount('/content/drive')

from google.colab import files
uploaded = files.upload()  # Upload your JSON file


# Authenticate with Hugging Face 
login("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") 

# Load the LLaMA 2  and tokenizer from Hugging Face
model_name = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = LlamaTokenizer.from_pretrained(model_name)
model = LlamaForCausalLM.from_pretrained(model_name)

#load data from a JSON file, each line in the file is treated as a separate JSON object
def load_json_data(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line.strip()))  # Load each line as a JSON object
    return data

# Define your JSON file path del
json_file_path = r'/content/qa_data.json'
data = load_json_data(json_file_path)

# Function to generate responses using Chain of Thought (CoT) structured prompting
def generate_response_with_cot(knowledge, question):
    # Construct the prompt with knowledge and question followed by a cue for an answer
    cot_prompt = f"{knowledge}\nQuestion: {question}\nAnswer based on the information above:"
    inputs = tokenizer(cot_prompt, return_tensors="pt")  # Tokenize the prompt for the model
    outputs = model.generate(**inputs, max_new_tokens=100)  # Generate response with a 100-token max limit
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)  # Decode the model's response
    return response

# Function to check if the model's answer is a hallucination by comparing it to the correct answer
def is_hallucination(model_answer, correct_answer):
    # Normalize answers by converting to lowercase and removing punctuation for comparison
    model_answer = re.sub(r'[^\w\s]', '', model_answer.lower())
    correct_answer = re.sub(r'[^\w\s]', '', correct_answer.lower())
    return correct_answer not in model_answer  # Returns True if correct answer isn't in model's answer

# Function to save results (question, answer, accuracy) to a JSON file
def save_results_to_json(results, output_file):
    with open(output_file, 'w') as file:
        json.dump(results, file, indent=4)

# Main function to generate responses and track accuracy and hallucination counts
def main_with_cot(json_file):
    data = load_json_data(json_file)
    hallucination_count = 0  # Counter for hallucinated answers
    correct_count = 0  # Counter for correct answers
    accuracy_results = []  # List to store accuracy over time
    question_responses = []  # List to store responses for each question

    # Loop over each item in the data, processing the first 50 items
    for idx, item in enumerate(data[:50]):  
        knowledge = item['knowledge']  # Background knowledge for the question
        question = item['question']  # The question to answer
        right_answer = item['right_answer']  # Expected correct answer

        print(f"Generating response for question {idx + 1}: {question}")

        # Generate the model's response using CoT
        cot_response = generate_response_with_cot(knowledge, question)

        # Extract the answer from the model's response by isolating the text after the answer cue
        final_answer = cot_response.split('Answer based on the information above:')[-1].strip().split('.')[0].strip()

        # Check if the answer is a hallucination 
        hallucination = is_hallucination(final_answer, right_answer)
        if hallucination:
            hallucination_count += 1  # Increment hallucination count
            evaluation = "Incorrect"
        else:
            correct_count += 1  # Increment correct count
            evaluation = "Correct"

        # Calculate accuracy after each question
        accuracy = (correct_count / (idx + 1)) * 100
        accuracy_results.append(accuracy)  # Append the current accuracy to the list

        # Store details of each question, including evaluation of correctness
        question_responses.append({
            "question": question,
            "generated_answer": final_answer,
            "correct_answer": right_answer,
            "evaluation": evaluation
        })

        # Print response and current stats for tracking progress
        print(f"CoT Response: {cot_response}\n")
        print(f"Model's Answer: {final_answer}\n")
        print(f"Hallucination Counter after question {idx + 1}: {hallucination_count}\n")
        print(f"Accuracy after question {idx + 1}: {accuracy}%\n")

    # Store and save all results in a JSON format for analysis
    results = {
        "total_correct_answers": correct_count,
        "total_hallucinated_answers": hallucination_count,
        "accuracy_results": accuracy_results,
        "question_responses": question_responses
    }
    output_file = "/content/drive/MyDrive/COT_results.json"
    save_results_to_json(results, output_file)

    # Plot the accuracy over time across the 50 questions
    plot_accuracy(accuracy_results)

# Function to plot accuracy across questions using matplotlib
def plot_accuracy(accuracy_results):
    plt.plot(range(1, len(accuracy_results) + 1), accuracy_results, marker='o')
    plt.xlabel('Question Number')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy Over Time')
    plt.grid(True)
    plt.show()

# Entry point of the script
if __name__ == "__main__":
    json_file = json_file_path  # Set the path of the JSON file
    main_with_cot(json_file)  # Run the main function with CoT processing

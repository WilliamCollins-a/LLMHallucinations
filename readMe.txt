This file is intended to run the methods we have coded to reroduce our results
All methods were run in google colab. A hugging face account is required for loading the Large language model.
such as Llama. data sets and methods can be found in the GitHub


//liabries to install
!pip install transformers sentencepiece huggingface_hub
!pip install bitsandbytes
!pip install sentence-transformers
!pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118
!pip install matplotlib
from huggingface_hub import login
login()

//mounts google drive to colab

from google.colab import drive
drive.mount('/content/drive')

// uploads data files to google drive
from google.colab import files
uploaded = files.upload()  # Upload your JSON file


// load the method to test. (e.g CoT)

import json
import re
import matplotlib.pyplot as plt
from transformers import LlamaForCausalLM, LlamaTokenizer
from huggingface_hub import login

# Authenticate with Hugging Face
login("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")  # Replace with your token

# Load the LLaMA 2 chat model and tokenizer
model_name = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = LlamaTokenizer.from_pretrained(model_name)
model = LlamaForCausalLM.from_pretrained(model_name)

# Load the data from a JSON file
def load_json_data(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line.strip()))  # Load each line as a JSON object
    return data

# Define your JSON file path
json_file_path = r'/content/qa_data.json'
data = load_json_data(json_file_path)

# Generate response with structured prompting
def generate_response_with_cot(knowledge, question):
    # Construct the prompt with a more structured question
    cot_prompt = f"{knowledge}\nQuestion: {question}\nAnswer based on the information above:"
    inputs = tokenizer(cot_prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=100)  # Limit to 100 tokens for concise responses
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Compare answers with normalization
def is_hallucination(model_answer, correct_answer):
    # Normalize for comparison
    model_answer = re.sub(r'[^\w\s]', '', model_answer.lower())
    correct_answer = re.sub(r'[^\w\s]', '', correct_answer.lower())
    return correct_answer not in model_answer

# Save results to a JSON file
def save_results_to_json(results, output_file):
    with open(output_file, 'w') as file:
        json.dump(results, file, indent=4)

# Main function to load data, generate responses, and track accuracy
def main_with_cot(json_file):
    data = load_json_data(json_file)
    hallucination_count = 0
    correct_count = 0
    accuracy_results = []
    question_responses = []

    for idx, item in enumerate(data[:50]):  # Process only the first 50 items
        knowledge = item['knowledge']
        question = item['question']
        right_answer = item['right_answer']

        print(f"Generating response for question {idx + 1}: {question}")

        # Generate the response using CoT
        cot_response = generate_response_with_cot(knowledge, question)

        # Extract the final answer from the response
        final_answer = cot_response.split('Answer based on the information above:')[-1].strip().split('.')[0].strip()

        # Determine if it's a hallucination
        hallucination = is_hallucination(final_answer, right_answer)
        if hallucination:
            hallucination_count += 1
            evaluation = "Incorrect"
        else:
            correct_count += 1
            evaluation = "Correct"

        # Calculate and store accuracy at each step
        accuracy = (correct_count / (idx + 1)) * 100
        accuracy_results.append(accuracy)

        # Store the question, answer, and evaluation for this question
        question_responses.append({
            "question": question,
            "generated_answer": final_answer,
            "correct_answer": right_answer,
            "evaluation": evaluation
        })

        print(f"CoT Response: {cot_response}\n")
        print(f"Model's Answer: {final_answer}\n")
        print(f"Hallucination Counter after question {idx + 1}: {hallucination_count}\n")
        print(f"Accuracy after question {idx + 1}: {accuracy}%\n")

    # Final results to save
    results = {
        "total_correct_answers": correct_count,
        "total_hallucinated_answers": hallucination_count,
        "accuracy_results": accuracy_results,
        "question_responses": question_responses
    }

    # Save results to JSON file
    output_file = "/content/drive/MyDrive/COT_results.json"
    save_results_to_json(results, output_file)

    # Plot accuracy over questions
    plot_accuracy(accuracy_results)

def plot_accuracy(accuracy_results):
    plt.plot(range(1, len(accuracy_results) + 1), accuracy_results, marker='o')
    plt.xlabel('Question Number')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy Over Time')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    json_file = json_file_path  # Use the defined path variable
    main_with_cot(json_file)
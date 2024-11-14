from huggingface_hub import login  # Import the Hugging Face login function
import json  # Import JSON module for data handling
import re  # Import regular expressions for text normalization
import matplotlib.pyplot as plt  # Import matplotlib for plotting results
from transformers import LlamaForCausalLM, LlamaTokenizer  # Import LLaMA model and tokenizer

# Authenticate with Hugging Face 
login("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") 

# Load the LLaMA 2 chat model and tokenizer
model_name = "meta-llama/Llama-2-7b-chat-hf"  
tokenizer = LlamaTokenizer.from_pretrained(model_name)  
model = LlamaForCausalLM.from_pretrained(model_name)  

# Function to load data from a JSON file
def load_json_data(file_path):
    data = []  # Initialize an empty list to hold the data
    with open(file_path, 'r') as f:  # Open the JSON file for reading
        for line in f:
            data.append(json.loads(line.strip()))  # Load each line as a JSON object and append to the list
    return data  

# Define  JSON file path
json_file_path = r'/content/qa_data.json' 
data = load_json_data(json_file_path)  

# Function to generate first response from the model
def generate_initial_response(knowledge, question):
    # Construct the prompt for the model
    initial_prompt = f"{knowledge}\nQuestion: {question}\nAnswer based on the information above (provide a concise answer):"
    inputs = tokenizer(initial_prompt, return_tensors="pt")  # Tokenize the prompt
    outputs = model.generate(**inputs, max_new_tokens=50)  # Generate a response, limiting to 50 tokens
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)  # Decode the output to text
    return response  

# Function for self-refining  initial response
def refine_response(knowledge, question, initial_answer):
    # Construct  refinement prompt
    refinement_prompt = f"{knowledge}\nQuestion: {question}\nInitial Answer: {initial_answer}\nPlease refine the answer to improve accuracy (provide a concise answer):"
    inputs = tokenizer(refinement_prompt, return_tensors="pt")  # Tokenize the refinement prompt
    outputs = model.generate(**inputs, max_new_tokens=20)  # Generate a refined response with a token limit
    refined_response = tokenizer.decode(outputs[0], skip_special_tokens=True)  # Decode the refined output
    return refined_response  

# Function to compare model answer with the correct answer and check for hallucinations
def is_hallucination(model_answer, correct_answer):
    # Normalize the answers for comparison (remove punctuation and convert to lowercase)
    model_answer = re.sub(r'[^\w\s]', '', model_answer.lower())
    correct_answer = re.sub(r'[^\w\s]', '', correct_answer.lower())
    return correct_answer not in model_answer  # Return True if the correct answer is not found in the model answer

# save results to a JSON file
def save_results_to_json(results, output_file):
    with open(output_file, 'w') as file:
        json.dump(results, file, indent=4)  

# Main function to load data, generate responses, and track accuracy
def main_with_self_refine(json_file):
    data = load_json_data(json_file)  # Load the data from the provided JSON file
    hallucination_count = 0  # Initialize count for hallucinations
    correct_count = 0  # Initialize count for correct answers
    accuracy_results = []  # List to track accuracy results over time
    question_responses = []  # List to store responses for each question

    for idx, item in enumerate(data[:50]):  # Process the first 50 items from the data
        knowledge = item['knowledge']  # Extract knowledge from the current item
        question = item['question']  # Extract the question
        right_answer = item['right_answer']  # Extract the correct answer

        print(f"Generating response for question {idx + 1}: {question}")  # Print current question number

        # Generate the initial response from the model
        initial_response = generate_initial_response(knowledge, question)
        # Extract the answer from the response
        initial_answer = initial_response.split("Answer based on the information above:")[-1].strip().split('.')[0].strip()

        # Initialize refined_answer with the initial_answer
        refined_answer = initial_answer  # Start by assuming the refined answer is the initial answer

        # Check if initial answer is correct before refining
        if initial_answer.lower() != right_answer.lower():
            # Perform self-refinement only if the initial answer is not correct
            refined_response = refine_response(knowledge, question, initial_answer)  # Get the refined answer
            refined_answer = refined_response.split("Answer based on the information above:")[-1].strip().split('.')[0].strip()

        # Determine if the refined answer is a hallucination
        hallucination = is_hallucination(refined_answer, right_answer)  
        if hallucination:
            hallucination_count += 1  # Increment hallucination count if the answer is incorrect
            evaluation = "Incorrect"  # Set evaluation status
        else:
            correct_count += 1  # Increment correct count if the answer is correct
            evaluation = "Correct"  # Set evaluation status

        # Calculate and store accuracy 
        accuracy = (correct_count / (idx + 1)) * 100  # Calculate accuracy percentage
        accuracy_results.append(accuracy)  

        # Store the question, initial and refined answers, and evaluation for this question
        question_responses.append({
            "question": question,
            "initial_answer": initial_answer,
            "refined_answer": refined_answer,
            "correct_answer": right_answer,
            "evaluation": evaluation
        })

        # Print the responses and statistics
        print(f"Initial Response: {initial_response}\n")
        print(f"Refined Answer: {refined_answer}\n")
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
    output_file = "/content/drive/MyDrive/SelfRefine_results.json"  # Define output file path
    save_results_to_json(results, output_file)  # Save results

    # Plot accuracy over questions
    plot_accuracy(accuracy_results)  # Call function to plot accuracy

# Function to plot accuracy results over the questions
def plot_accuracy(accuracy_results):
    plt.plot(range(1, len(accuracy_results) + 1), accuracy_results, marker='o')  # Create a line plot of accuracy
    plt.xlabel('Question Number')  # Label for the x-axis
    plt.ylabel('Accuracy (%)')  # Label for the y-axis
    plt.title('Accuracy Over Time')  # Title for the plot
    plt.grid(True)  # Add grid for better readability
    plt.show()  # Display the plot

# Entry point for the script
if __name__ == "__main__":
    json_file = json_file_path  
    main_with_self_refine(json_file)  # Call the main function with the JSON file
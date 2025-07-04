from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Chatbot:
    def __init__(self):
        # Initialize the model and tokenizer
        self.model_name = "gpt2"  # You can also use "microsoft/DialoGPT-medium" for better chat performance
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        
        # Set device (GPU if available, else CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Chat history
        self.chat_history = []
        
        # Set generation parameters
        self.max_length = 200  # Increased from 100 to 200 for longer responses
        self.num_return_sequences = 1
        self.temperature = 0.7
        self.top_k = 50
        self.top_p = 0.95
        
    def generate_response(self, user_input):
        try:
            # Add user input to chat history
            self.chat_history.append(user_input)
            
            # Prepare the input
            input_text = " ".join(self.chat_history[-5:])  # Use last 5 messages for context
            inputs = self.tokenizer.encode(input_text, return_tensors="pt").to(self.device)
            
            # Generate response
            outputs = self.model.generate(
                inputs,
                max_length=self.max_length,
                num_return_sequences=self.num_return_sequences,
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            # Decode and clean the response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the new part of the response
            response = response[len(input_text):].strip()
            
            # Add response to chat history
            self.chat_history.append(response)
            
            # Keep chat history manageable
            if len(self.chat_history) > 10:
                self.chat_history = self.chat_history[-10:]
            
            return response
            
        except Exception as e:
            return f"An error occurred: {str(e)}"
    
    def clear_history(self):
        """Clear the chat history"""
        self.chat_history = []

def main():
    print("Welcome to the AI Chatbot!")
    print("Type 'quit' to exit the chat.")
    print("Type 'clear' to clear chat history.")
    print("-" * 50)
    
    # Create chatbot instance
    chatbot = Chatbot()
    
    while True:
        # Get user input
        user_input = input("\nYou: ").strip()
        
        # Check if user wants to quit
        if user_input.lower() == 'quit':
            print("\nThank you for using the AI Chatbot!")
            break
            
        # Check if user wants to clear history
        if user_input.lower() == 'clear':
            chatbot.clear_history()
            print("\nChat history cleared!")
            continue
        
        try:
            # Get response from the model
            response = chatbot.generate_response(user_input)
            print("\nBot:", response)
            
        except Exception as e:
            print(f"\nAn error occurred: {str(e)}")
            print("Please try again or type 'quit' to exit.")

if __name__ == "__main__":
    main() 
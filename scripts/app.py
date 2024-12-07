from llama_index.llms.ollama import Ollama
import streamlit as st
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import re
import pickle

class SentimentModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, dropout_rate=0.1, l2_lambda=0.001, num_classes=3):
        super(SentimentModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(embedding_dim * max_length, 128)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = self.flatten(x)
        x = nn.ReLU()(self.fc1(x))
        x = self.dropout1(x)
        x = nn.ReLU()(self.fc2(x))
        x = self.dropout2(x)
        x = nn.ReLU()(self.fc3(x))
        x = self.fc4(x)
        return x

# Load the vocabulary
with open(r"D:\Senti_chatbot\notebook\vocab.pkl", 'rb') as f:
    vocab = pickle.load(f)

# Define the preprocessing functions
max_length = 50  # Ensure this matches the value in training

def tokenize(text):
    return text.split()

def text_to_sequence(text, vocab, max_length):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)  # Remove non-alphabetic characters
    tokens = tokenize(text)
    sequence = [vocab.get(token, vocab["<unk>"]) for token in tokens]
    return sequence[:max_length] + [vocab["<pad>"]] * (max_length - len(sequence))

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SentimentModel(vocab_size=len(vocab))  # Use the length of the loaded vocab
model.load_state_dict(torch.load(r'D:\Senti_chatbot\notebook\sentiment_model.pth', map_location=device, weights_only=True))
model = model.to(device)
model.eval()

# Define the label mapping
label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}  # Ensure these match your training labels

# Prediction function
def predict_sentiment(sentence):
    sequence = text_to_sequence(sentence, vocab, max_length)
    sequence_tensor = torch.tensor([sequence]).to(device)
    with torch.no_grad():
        output = model(sequence_tensor)
        predicted_label = output.argmax(1).item()
    return label_map[predicted_label]


# Function to fetch a response from the Llama 3.1:1b model
def fetch_llama_response(prompt):
        # Initialize the Ollama model
    llama_model = Ollama(base_url="http://127.0.0.1:11436", model="llama3.2:1b")

        
        # Attempt to fetch response
    response_stream = llama_model.complete(prompt=prompt)
    return response_stream
    


# WhatsApp-style chatbot UI with Streamlit
def main():
    st.set_page_config(page_title="WhatsApp-Style Chatbot", layout="centered")
    st.markdown("<h1 style='text-align: center;'>🧠 Sentiment-Driven Chatbot</h1>", unsafe_allow_html=True)
    st.write("This chatbot tailors responses based on detected sentiment. Chat in a WhatsApp-style interface!")

    # Chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Input area
    user_input = st.text_input("You: ", placeholder="Type something...", key="user_input")
    if st.button("Send"):
        if user_input.strip():
            # Predict sentiment
            input_sentence = user_input
            sentiment = predict_sentiment(input_sentence)

            if "Error" in sentiment:
                st.warning("Sentiment prediction failed. Please try again.")
                return

            # Enhance the prompt with sentiment
            enhanced_prompt = f"The user's sentiment is {sentiment}. Respond empathetically to: {user_input}"
            bot_response = fetch_llama_response(enhanced_prompt)

            # Append messages
            st.session_state.messages.append({"sender": "user", "text": user_input, "sentiment": sentiment})
            st.session_state.messages.append({"sender": "bot", "text": bot_response, "sentiment": sentiment})
        else:
            st.warning("Please type something to interact with the chatbot.")

    # Display messages in WhatsApp-style
    for msg in st.session_state.messages:
        if msg["sender"] == "user":
            st.markdown(
                f"<div style='background-color: #0000ff; color: white; border-radius: 10px; padding: 10px; margin: 5px 20px 5px auto; text-align: right; max-width: 70%;'>"
                f"<strong>You:</strong> {msg['text']}<br><em>Sentiment: {msg['sentiment']}</em></div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"<div style='background-color: #66cc00; color: white; border-radius: 10px; padding: 10px; margin: 5px auto 5px 20px; text-align: left; max-width: 70%;'>"
                f"<strong>Bot:</strong> {msg['text']}</div>",
                unsafe_allow_html=True,
            )

if __name__ == "__main__":
    main()


import torch
import torch.nn as nn
import re
import pickle

# Define the SentimentModel class
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
model.load_state_dict(torch.load(r'D:\Senti_chatbot\notebook\sentiment_model.pth', map_location=device))
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

# Test the model with a sample sentence
input_sentence = "What is the time"
predicted_sentiment = predict_sentiment(input_sentence)
print(f"Sentence: {input_sentence}")
print(f"Predicted Sentiment: {predicted_sentiment}")

# 🧠 Sentiment-Driven Chatbot

This project is a sentiment-driven chatbot application powered by a PyTorch-based sentiment analysis model and the Ollama Llama model for generating empathetic responses. The chatbot runs on a Streamlit-based WhatsApp-style user interface.

## 🎥 Demo Video 
Here is a demo video of the sentiment-driven chatbot in action: 
[Demo Video](demo_video/senti_chatbot_video.mp4)


## 🔧 Project Setup

### 1. Prerequisites

- Install Python 3.8 or higher.
- Install the required dependencies using the command:
  ```bash
  pip install -r requirements.txt
## 2. Files in the Project  

- **sentiment_model.pth**: A PyTorch-based classification model trained for sentiment analysis (Positive, Neutral, Negative).  
- **vocab.pkl**: A vocabulary file mapping words to indices, used to preprocess user input for sentiment analysis.  

## 3. Llama Model Installation  

- Download and install Ollama, a locally hosted Llama model environment.  
- Run the following command to download the `llama3.2:1b` model:  
  ```bash  
  ollama run llama3.2:1b
- Ensure the Ollama server is running on http://127.0.0.1:11436.

## 4. Start the Application  

- To start the Streamlit application, run:  
  ```bash  
  streamlit run app.py

## 🛠 Project Details  

### 1. Sentiment Analysis  

The `sentiment_model.pth` is a PyTorch model that classifies user input into one of three sentiment categories:  

- Positive  
- Neutral  
- Negative  

The model is trained using a sequence of words converted to indices from the `vocab.pkl` file. User input is preprocessed, tokenized, and padded to a maximum length before passing through the model.  

### 1. Llama for Response Generation  

The chatbot generates empathetic responses based on user sentiment using the `llama3.2:1b` model. The sentiment information is appended to the user input to create a prompt that guides the Llama model.  

**Example Prompt:**
The user's sentiment is Positive. Respond empathetically to: "What a wonderful day!"

The response is fetched using the Ollama API hosted locally.

### 3. User Interface  

The application uses a WhatsApp-style chat interface created with Streamlit. Features include:  

- User messages displayed on the right in blue.  
- Chatbot responses displayed on the left in green.  
- Sentiment information displayed alongside user messages.  

## 📚 Dependencies  

The project uses the following Python libraries:  

- **Streamlit**: For creating the chat interface.  
- **Torch**: For the sentiment analysis model.  
- **Ollama API**: To interact with the Llama model for response generation.  

- Install all dependencies using:  
  ```bash  
  pip install -r requirements.txt

## 🚀 How to Use  

1. Start the Ollama server with the `llama3.2:1b` model:  
   ```bash  
   ollama run llama3.2:1b
2. Run the chatbot application:
   ```bash
   streamlit run app.py
3. Interact with the chatbot in the browser at [http://localhost:8501](http://localhost:8501).  

## ✨ Features  

- Sentiment-based empathetic response generation.  
- Intuitive WhatsApp-style chat interface.  
- Locally hosted Llama model for privacy and performance.  

## 📝 Notes  

- Ensure the Ollama server is running before starting the chatbot application.  
- Update the port ([http://127.0.0.1:11436](http://127.0.0.1:11436)) in the `app.py` file if using a different port for Ollama.

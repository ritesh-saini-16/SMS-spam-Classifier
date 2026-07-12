# Email/SMS Spam Classifier

A machine learning project that classifies emails and SMS messages as spam or not spam using Natural Language Processing (NLP) and Machine Learning.

## Features
- Text preprocessing using NLTK
- TF-IDF vectorization
- Multinomial Naive Bayes classification
- Interactive web interface using Streamlit

## Live Demo
You can try the spam classifier at: http://localhost:8501

## GitHub Repository
Visit the project on GitHub: [SMS-spam-Classifier](https://github.com/ritesh-saini-16/SMS-spam-Classifier)

## Project Structure
- `app.py`: Streamlit web application
- `train_model.py`: Script to train and save the model
- `spam.csv`: Dataset containing labeled SMS messages
- `vectorizer.pkl`: Saved TF-IDF vectorizer
- `model.pkl`: Saved trained model

## Setup and Installation
1. Clone the repository:
   ```
   git clone https://github.com/ritesh-saini-16/SMS-spam-Classifier.git
   cd SMS-spam-Classifier
   ```
2. Install required packages:
   ```
   pip install -r requirements.txt
   ```
3. Run the training script:
   ```
   python train_model.py
   ```
4. Launch the web app:
   ```
   streamlit run app.py
   ```

## Technologies Used
- Python
- NLTK
- scikit-learn
- Streamlit
- pandas
- numpy

## Usage
1. Enter any text message in the text area
2. Click the "Predict" button
3. The app will classify the message as either "Spam" or "Not Spam"

## Contributing
Feel free to fork this repository and submit pull requests for any improvements.

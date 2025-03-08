import pandas as pd
import nltk

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics.pairwise import cosine_similarity
from chatterbot import ChatBot
from chatterbot.trainers import ListTrainer

def load_books_dataset(file_path):
    df = pd.read_csv(file_path, encoding='latin1', delimiter=';')
    df = df[['Book-Title', 'Book-Author', 'Year-Of-Publication', 'Publisher']]
    df.dropna(inplace=True)
    return df

def train_recommendation_model(df):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['Book-Title'])
    return tfidf, tfidf_matrix

def recommend_books(title, df, tfidf, tfidf_matrix, top_n=5):
    query_vec = tfidf.transform([title])
    similarity = cosine_similarity(query_vec, tfidf_matrix).flatten()
    indices = similarity.argsort()[-top_n:][::-1]
    return df.iloc[indices][['Book-Title', 'Book-Author']].values.tolist()

def train_chatbot():
    chatbot = ChatBot('BookBot')
    trainer = ListTrainer(chatbot)
    trainer.train([
        "Can you recommend a book?",
        "Sure! What genre or book are you interested in?",
        "I like mystery books.",
        "I recommend 'The Girl with the Dragon Tattoo' by Stieg Larsson."
    ])
    return chatbot

def main():
    file_path = '/content/drive/MyDrive/A/preprocessed_data.csv'  
    df = load_books_dataset(file_path)
    tfidf, tfidf_matrix = train_recommendation_model(df)
    chatbot = train_chatbot()

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break
        response = chatbot.get_response(user_input)
        print(f"BookBot: {response}")

        recommended = recommend_books(user_input, df, tfidf, tfidf_matrix)
        if recommended:
            print("Here are some book recommendations:")
            for book, author in recommended:
                print(f"- {book} by {author}")

if __name__ == "__main__":
    main()
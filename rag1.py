# Import necessary libraries
import pandas as pd
import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from sentence_transformers import SentenceTransformer
import cratedb
import openai

# Load hotel data and preprocess
hotel_data = pd.read_csv('hotel_data.csv')
hotel_data['description'] = hotel_data['description'].strip()
hotel_data['reviews'] = hotel_data['reviews'].strip()

openai.api_key = ""

# Embed hotel data using Sentence-BERT
model = SentenceTransformer('all-MiniLM-L6-v2')
hotel_vectors = model.encode(hotel_data[['description', 'reviews']].agg(', '.join, axis=1))

# Connect to CrateDB and create a table for hotel vectors
client = cratedb.Client(username='<username>', password='<password>', host='<host>', port=<port>)
client.execute("CREATE TABLE hotel_vectors (id SERIAL PRIMARY KEY, vector FLOAT_VECTOR(768))")

# Insert hotel vectors into CrateDB
for i, vector in enumerate(hotel_vectors):
    client.execute("INSERT INTO hotel_vectors (vector) VALUES ($1)", (vector,))

# Define a function for semantic search
def semantic_search(query, k=5):
    query_vector = model.encode([query])
    knn = NearestNeighbors(n_neighbors=k, metric='euclidean').fit(hotel_vectors)
    distances, indices = knn.kneighbors(query_vector)
    return hotel_data.iloc[indices[0]]

# Define a function for generating responses
def generate_response(hotels, query):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",# Load pre-trained language model
        messages=f"Here are some hotels that match your preferences: {hotels.to_string(index=False)}\n\n"
                 f"Why do these hotels match {query}?",
        temperature=0.7,
        max_tokens=150,
        top_p=1,
        frequency_penalty=0.3,
        presence_penalty=0.5,
    )
    return response.strip()

# Main function
def main():
    while True:
        query = input("Enter your semantic query: ")
        hotels = semantic_search(query)
        response = generate_response(hotels, query)
        print(response)

if __name__ == "__main__":
    main()
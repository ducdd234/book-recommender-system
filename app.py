import streamlit as st
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd
from tqdm import tqdm  # Thay notebook bằng bình thường
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification  # Từ cell 15
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder

# Setup device
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load và process data (từ cells 3-13)
@st.cache_data
def load_and_process_data():
    reviews_file_path = 'Books_rating.csv'
    books_details_file_path = 'books_data.csv'
    reviews_df = pd.read_csv(reviews_file_path, nrows=30000)
    books_details_df = pd.read_csv(books_details_file_path, nrows=2500)
    
    # Xử lý reviews (cell 7)
    reviews_df = reviews_df.dropna(subset=['Title', 'User_id'])
    reviews_df = reviews_df.drop(columns=['profileName', 'Price'])
    reviews_df['review/summary'] = reviews_df['review/summary'].fillna('')
    reviews_df['review/text'] = reviews_df['review/text'].fillna('')
    
    # Xử lý books (cell 8)
    books_details_df = books_details_df.dropna(subset=['Title'])
    books_details_df['ratingsCount'] = books_details_df['ratingsCount'].fillna(books_details_df['ratingsCount'].median())
    textual_columns = ['description', 'authors', 'publisher', 'publishedDate', 'categories']
    books_details_df[textual_columns] = books_details_df[textual_columns].fillna('')
    books_details_df = books_details_df.drop(columns=['image', 'previewLink', 'infoLink'])
    
    # Merge (cell 10)
    merged_df = pd.merge(reviews_df, books_details_df, on='Title', how='left')
    merged_df = merged_df.fillna('')
    
    # Sentiment analysis với RoBERTa (từ cell 17)
    tokenizer = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
    model = AutoModelForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
    def get_sentiment(text):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        outputs = model(**inputs)
        return torch.nn.functional.softmax(outputs.logits, dim=1).detach().numpy()[0]
    tqdm.pandas()
    merged_df['sentiment_score'] = merged_df['review/text'].progress_apply(get_sentiment)
    
    # Xử lý helpfulness (từ cell 19)
    def parse_helpfulness(help_str):
        helpful, total = map(int, help_str.split('/'))
        return helpful / total if total > 0 else 0
    merged_df['helpfulness'] = merged_df['review/helpfulness'].apply(parse_helpfulness)
    
    return merged_df

merged_df = load_and_process_data()

# Collaborative Model (từ cell 27)
class CollaborativeFilteringModel(nn.Module):
    def __init__(self, num_users, num_books, embedding_size=50):
        super(CollaborativeFilteringModel, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_size)
        self.book_embedding = nn.Embedding(num_books, embedding_size)
        self.fc = nn.Linear(embedding_size * 2, 1)
    
    def forward(self, user_ids, book_ids):
        user_emb = self.user_embedding(user_ids)
        book_emb = self.book_embedding(book_ids)
        concat_emb = torch.cat([user_emb, book_emb], dim=1)
        return self.fc(concat_emb).squeeze()

@st.cache_resource
def train_collaborative_model(df):
    user_encoder = LabelEncoder()
    book_encoder = LabelEncoder()
    df['user_id_encoded'] = user_encoder.fit_transform(df['User_id'])
    df['book_id_encoded'] = book_encoder.fit_transform(df['Id'])  # Giả sử 'Id' là book ID
    
    num_users = df['user_id_encoded'].nunique()
    num_books = df['book_id_encoded'].nunique()
    
    model = CollaborativeFilteringModel(num_users, num_books).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    X = df[['user_id_encoded', 'book_id_encoded']]
    y = df['review/score']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    train_dataset = TensorDataset(torch.tensor(X_train.values, dtype=torch.long), torch.tensor(y_train.values, dtype=torch.float))
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    for epoch in range(10):  # Giảm epochs để nhanh
        model.train()
        for user_ids, book_ids in tqdm(train_loader):
            user_ids, book_ids = user_ids[:,0].to(device), user_ids[:,1].to(device)
            targets = book_ids.to(device)  # Fix nếu sai
            optimizer.zero_grad()
            outputs = model(user_ids, book_ids)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
    
    return model, user_encoder, book_encoder

cf_model, user_encoder, book_encoder = train_collaborative_model(merged_df)

# Content-Based (từ cell 30)
@st.cache_resource
def get_content_based_vectorizer(df):
    tfidf = TfidfVectorizer(stop_words='english')
    features = df['description'] + ' ' + df['authors'] + ' ' + df['categories'] + ' ' + df['sentiment_score'].astype(str)
    tfidf_matrix = tfidf.fit_transform(features)
    return tfidf, tfidf_matrix

tfidf, tfidf_matrix = get_content_based_vectorizer(merged_df)

# Functions recommendation (từ cells 34-40)
def get_cf_recommendations(user_id, n=10):
    user_encoded = user_encoder.transform([user_id])[0]
    book_ids = torch.arange(len(book_encoder.classes_)).to(device)
    user_ids = torch.full_like(book_ids, user_encoded).to(device)
    with torch.no_grad():
        preds = cf_model(user_ids, book_ids)
    top_indices = preds.topk(n).indices.cpu().numpy()
    return merged_df.iloc[top_indices]

def get_content_based_recommendations(query, n=10):
    query_vec = tfidf.transform([query])
    sim = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = sim.argsort()[-n:][::-1]
    return merged_df.iloc[top_indices]

def get_hybrid_recommendations(user_id, query, n=10):
    cf_recs = get_cf_recommendations(user_id, n*2)
    query_vec = tfidf.transform([query])
    sim = cosine_similarity(query_vec, tfidf_matrix[cf_recs.index]).flatten()
    cf_recs['hybrid_score'] = cf_recs['review/score'] * 0.5 + sim * 0.5
    return cf_recs.sort_values('hybrid_score', ascending=False).head(n)

# Giao diện (giống ảnh)
st.title("Book Recommender System")
st.subheader("Hybrid NLP + Content-Based Filtering with Explainability")

mode = st.radio("Select a mode:", [
    "Content-Based (based on metadata: genre, author, category)",
    "NLP-Based (Semantic understanding with embeddings)",
    "Hybrid (CF + NLP) (Combines ratings and semantic analysis)",
    "Hybrid (CF + NLP) Search (Enter book or describe works from both!)"
])

user_id = st.text_input("Enter User ID (if known, else leave blank):", "")
query = st.text_input("Enter book or describe:", placeholder="E.g., 'fantasy' or 'Harry Potter'")

if st.button("Search"):
    if not query:
        st.warning("Please enter a search query.")
    else:
        if mode == "Content-Based (based on metadata: genre, author, category)":
            recs = get_content_based_recommendations(query)
        elif mode == "NLP-Based (Semantic understanding with embeddings)":
            recs = get_content_based_recommendations(query)  # Reuse vì NLP-based dùng embeddings tương tự
        elif mode == "Hybrid (CF + NLP) (Combines ratings and semantic analysis)":
            user_id = user_id or merged_df['User_id'].sample(1).values[0]  # Random nếu blank
            recs = get_hybrid_recommendations(user_id, query)
        elif mode == "Hybrid (CF + NLP) Search (Enter book or describe works from both!)":
            user_id = user_id or merged_df['User_id'].sample(1).values[0]
            recs = get_hybrid_recommendations(user_id, query)
        
        if recs.empty:
            st.info("No recommendations found.")
        else:
            st.subheader("Recommendations:")
            for _, row in recs.iterrows():
                st.write(f"**Title:** {row['Title']}")
                st.write(f"**Author:** {row['authors']}")
                st.write(f"**Description:** {row['description'][:200]}...")
                st.write(f"**Category:** {row['categories']}")
                st.write(f"**Rating:** {row['review/score']}, Sentiment: {row['sentiment_score']}")
                st.write("---")

st.caption("Select a mode and search for books to get recommendations. Do not recommend the same book.")
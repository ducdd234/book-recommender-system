import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# ============= CẤU HÌNH TRANG =============
st.set_page_config(
    page_title="Book Recommender System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============= CSS STYLING =============
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .method-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .book-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 3px solid #667eea;
    }
    .stButton>button {
        width: 100%;
        background: #667eea;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        font-weight: 600;
    }
    .stButton>button:hover {
        background: #764ba2;
    }
</style>
""", unsafe_allow_html=True)

# ============= ĐỊNH NGHĨA MODEL NCF =============
class NCF_Model(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=32):
        super(NCF_Model, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim * 2, 64)
        self.fc2 = nn.Linear(64, 32)
        self.output = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, user_idx, item_idx):
        u_emb = self.user_embedding(user_idx)
        i_emb = self.item_embedding(item_idx)
        x = torch.cat([u_emb, i_emb], dim=1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.relu(self.fc2(x))
        x = self.output(x)
        return torch.sigmoid(x) * 4 + 1

# ============= LOAD DỮ LIỆU VÀ MODEL =============
@st.cache_resource
def load_models_and_data():
    """Load tất cả models và data cần thiết"""
    try:
        # Load CSV files
        merged_df = pd.read_csv('step1_completed.csv')
        
        # Load pickle file nếu có
        try:
            with open('recommendation_models.pkl', 'rb') as f:
                saved_data = pickle.load(f)
            
            # Khôi phục CF model
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            num_users = len(saved_data['user_to_index'])
            num_items = len(saved_data['item_to_index'])
            
            model_cf = NCF_Model(num_users, num_items).to(device)
            model_cf.load_state_dict(saved_data['model_cf'])
            model_cf.eval()
            
            # Load SBERT model
            sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            return {
                'merged_df': merged_df,
                'model_cf': model_cf,
                'user_to_index': saved_data['user_to_index'],
                'item_to_index': saved_data['item_to_index'],
                'index_to_item': saved_data['index_to_item'],
                'book_vectors': saved_data['book_vectors'],
                'book_title_to_idx': saved_data['book_title_to_idx'],
                'final_book_data': saved_data['final_book_data'],
                'sbert_model': sbert_model,
                'device': device
            }
        except FileNotFoundError:
            st.error("⚠️ Không tìm thấy file 'recommendation_models.pkl'. Vui lòng chạy notebook để tạo file này!")
            return None
            
    except Exception as e:
        st.error(f"Lỗi khi load dữ liệu: {str(e)}")
        return None

# ============= HÀM GỢI Ý CONTENT-BASED =============
def recommend_content_based(book_title, data, top_k=5):
    """Gợi ý dựa trên metadata"""
    try:
        merged_df = data['merged_df']
        
        # Tìm sách
        book_data = merged_df[merged_df['Title'].str.contains(book_title, case=False, na=False)]
        
        if book_data.empty:
            return pd.DataFrame()
        
        # Lấy thông tin sách
        target_categories = book_data['categories'].iloc[0]
        target_authors = book_data['authors'].iloc[0]
        
        # Tìm sách tương tự
        similar_books = merged_df[
            (merged_df['categories'] == target_categories) |
            (merged_df['authors'] == target_authors)
        ]
        
        # Loại bỏ sách gốc
        similar_books = similar_books[similar_books['Title'] != book_data['Title'].iloc[0]]
        
        # Lấy top k
        result = similar_books.groupby('Title').agg({
            'review/score': 'mean',
            'authors': 'first',
            'categories': 'first'
        }).reset_index()
        
        result = result.sort_values('review/score', ascending=False).head(top_k)
        result.columns = ['Title', 'Avg Rating', 'Authors', 'Categories']
        
        return result
        
    except Exception as e:
        st.error(f"Lỗi content-based: {str(e)}")
        return pd.DataFrame()

# ============= HÀM GỢI Ý NLP-BASED =============
def recommend_nlp_based(query, data, top_k=5):
    """Gợi ý dựa trên semantic similarity"""
    try:
        sbert_model = data['sbert_model']
        book_vectors = data['book_vectors']
        book_title_to_idx = data['book_title_to_idx']
        final_book_data = data['final_book_data']
        
        # Tạo embedding cho query
        query_vector = sbert_model.encode([query])
        
        # Tính similarity
        similarities = cosine_similarity(query_vector, book_vectors)[0]
        
        # Lấy top k
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            book_title = final_book_data.iloc[idx]['Title']
            sim_score = similarities[idx]
            impact = final_book_data.iloc[idx]['final_impact_score']
            
            results.append({
                'Title': book_title,
                'Similarity': f"{sim_score:.3f}",
                'Impact Score': f"{impact:.2f}"
            })
        
        return pd.DataFrame(results)
        
    except Exception as e:
        st.error(f"Lỗi NLP-based: {str(e)}")
        return pd.DataFrame()

# ============= HÀM GỢI Ý HYBRID =============
def recommend_hybrid(user_id, data, top_k=5, w_cf=0.5, w_content=0.3, w_impact=0.2):
    """Gợi ý hybrid kết hợp CF + Content"""
    try:
        merged_df = data['merged_df']
        model_cf = data['model_cf']
        user_to_index = data['user_to_index']
        item_to_index = data['item_to_index']
        book_vectors = data['book_vectors']
        book_title_to_idx = data['book_title_to_idx']
        final_book_data = data['final_book_data']
        device = data['device']
        
        # Kiểm tra user
        if user_id not in user_to_index:
            st.warning(f"User ID '{user_id}' không tồn tại trong hệ thống!")
            return pd.DataFrame()
        
        # Lấy sách chưa đọc
        read_books = merged_df[merged_df['User_id'] == user_id]['Title'].unique()
        all_books = final_book_data['Title'].unique()
        unread_books = list(set(all_books) - set(read_books))
        
        if len(unread_books) == 0:
            st.info("User đã đọc hết sách!")
            return pd.DataFrame()
        
        # Sample candidates
        n_sample = min(len(unread_books), 500)
        candidates = np.random.choice(unread_books, n_sample, replace=False)
        
        # Tính user profile
        liked_books = merged_df[(merged_df['User_id'] == user_id) & 
                               (merged_df['review/score'] >= 4)]['Title'].unique()
        
        user_profile_vec = None
        if len(liked_books) > 0:
            valid_indices = [book_title_to_idx[t] for t in liked_books if t in book_title_to_idx]
            if valid_indices:
                user_profile_vec = np.mean(book_vectors[valid_indices], axis=0)
        
        # Tính CF predictions
        u_idx = user_to_index[user_id]
        u_tensor = torch.tensor([u_idx] * len(candidates), dtype=torch.long).to(device)
        i_tensor = torch.tensor([item_to_index.get(t, 0) for t in candidates], 
                               dtype=torch.long).to(device)
        
        with torch.no_grad():
            cf_preds = model_cf(u_tensor, i_tensor).squeeze().cpu().numpy()
            if cf_preds.ndim == 0:
                cf_preds = np.array([cf_preds])
        
        # Tính hybrid scores
        results = []
        for i, title in enumerate(candidates):
            score_cf = cf_preds[i] / 5.0
            
            # Content similarity
            score_content = 0
            if user_profile_vec is not None and title in book_title_to_idx:
                book_vec = book_vectors[book_title_to_idx[title]]
                score_content = np.dot(user_profile_vec, book_vec) / (
                    np.linalg.norm(user_profile_vec) * np.linalg.norm(book_vec) + 1e-9
                )
            
            # Impact score
            score_impact = 0
            raw_impact = 0
            if title in book_title_to_idx:
                raw_impact = final_book_data.loc[book_title_to_idx[title], 'final_impact_score']
                score_impact = min(max(raw_impact / 5.0, 0), 1)
            
            final_score = (score_cf * w_cf) + (score_content * w_content) + (score_impact * w_impact)
            
            results.append({
                'Title': title,
                'Hybrid Score': final_score,
                'CF Pred': score_cf * 5,
                'Content Sim': score_content,
                'Impact': raw_impact
            })
        
        df_result = pd.DataFrame(results).sort_values('Hybrid Score', ascending=False).head(top_k)
        return df_result
        
    except Exception as e:
        st.error(f"Lỗi hybrid: {str(e)}")
        return pd.DataFrame()

# ============= HÀM GỢI Ý HYBRID SEARCH =============
def recommend_hybrid_search(query, data, top_k=5):
    """Gợi ý hybrid dựa trên query (không cần user_id)"""
    try:
        sbert_model = data['sbert_model']
        book_vectors = data['book_vectors']
        book_title_to_idx = data['book_title_to_idx']
        final_book_data = data['final_book_data']
        merged_df = data['merged_df']
        
        # Tạo embedding cho query
        query_vector = sbert_model.encode([query])
        
        # Tính similarity với tất cả sách
        similarities = cosine_similarity(query_vector, book_vectors)[0]
        
        # Lấy top candidates
        top_indices = similarities.argsort()[-50:][::-1]
        
        results = []
        for idx in top_indices[:top_k]:
            title = final_book_data.iloc[idx]['Title']
            sim_score = similarities[idx]
            impact = final_book_data.iloc[idx]['final_impact_score']
            
            # Lấy rating trung bình
            book_ratings = merged_df[merged_df['Title'] == title]['review/score']
            avg_rating = book_ratings.mean() if len(book_ratings) > 0 else 0
            
            # Tính hybrid score
            hybrid_score = (sim_score * 0.5) + (impact / 5.0 * 0.3) + (avg_rating / 5.0 * 0.2)
            
            results.append({
                'Title': title,
                'Hybrid Score': hybrid_score,
                'Similarity': sim_score,
                'Impact': impact,
                'Avg Rating': avg_rating
            })
        
        df_result = pd.DataFrame(results).sort_values('Hybrid Score', ascending=False)
        return df_result
        
    except Exception as e:
        st.error(f"Lỗi hybrid search: {str(e)}")
        return pd.DataFrame()

# ============= MAIN APP =============
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>📚 Book Recommender System</h1>
        <p>Hybrid NLP + Content-Based Filtering with Explainability</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data
    data = load_models_and_data()
    
    if data is None:
        st.stop()
    
    # ========== CONTENT-BASED ==========
    st.markdown("""
    <div class="method-card">
        <h2>🔵 Content-Based</h2>
        <p>Based on metadata (genre, author, category)</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 1])
    with col1:
        book_input = st.text_input("Enter book title:", 
                                   placeholder="e.g., Harry Potter",
                                   key="content_input")
    with col2:
        st.write("")
        st.write("")
        search_btn1 = st.button("🔍 Search", key="content_btn")
    
    if search_btn1 and book_input:
        with st.spinner("Searching..."):
            results = recommend_content_based(book_input, data, top_k=5)
            
            if not results.empty:
                st.success(f"Found {len(results)} similar books!")
                st.dataframe(results, use_container_width=True)
            else:
                st.warning("No books found. Try another title!")
    
    st.markdown("---")
    
    # ========== NLP-BASED ==========
    st.markdown("""
    <div class="method-card">
        <h2>⚡ NLP-Based</h2>
        <p>Semantic understanding with embeddings</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 1])
    with col1:
        query_input = st.text_input("Enter book or description:", 
                                    placeholder="Works with both!",
                                    key="nlp_input")
    with col2:
        st.write("")
        st.write("")
        search_btn2 = st.button("🔍 Search", key="nlp_btn")
    
    if search_btn2 and query_input:
        with st.spinner("Analyzing semantics..."):
            results = recommend_nlp_based(query_input, data, top_k=5)
            
            if not results.empty:
                st.success(f"Found {len(results)} relevant books!")
                st.dataframe(results, use_container_width=True)
            else:
                st.warning("No results found!")
    
    st.markdown("---")
    
    # ========== HYBRID (CF + NLP) ==========
    st.markdown("""
    <div class="method-card">
        <h2>🟢 Hybrid (CF + NLP)</h2>
        <p>Combines ratings and semantic analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    # User ID input
    unique_users = list(data['user_to_index'].keys())
    sample_users = unique_users[:100]  # Hiển thị 100 users đầu
    
    col1, col2 = st.columns([3, 1])
    with col1:
        user_id_input = st.selectbox("Select User ID:", 
                                     options=sample_users,
                                     key="hybrid_user")
    with col2:
        st.write("")
        st.write("")
        search_btn3 = st.button("🔍 Get Recommendations", key="hybrid_btn")
    
    if search_btn3 and user_id_input:
        with st.spinner("Generating personalized recommendations..."):
            results = recommend_hybrid(user_id_input, data, top_k=5)
            
            if not results.empty:
                st.success(f"Top 5 recommendations for user {user_id_input}:")
                
                # Format kết quả đẹp hơn
                display_df = results[['Title', 'Hybrid Score', 'CF Pred', 'Content Sim', 'Impact']].copy()
                display_df['Hybrid Score'] = display_df['Hybrid Score'].apply(lambda x: f"{x:.3f}")
                display_df['CF Pred'] = display_df['CF Pred'].apply(lambda x: f"{x:.2f}/5")
                display_df['Content Sim'] = display_df['Content Sim'].apply(lambda x: f"{x:.3f}")
                display_df['Impact'] = display_df['Impact'].apply(lambda x: f"{x:.2f}")
                
                st.dataframe(display_df, use_container_width=True)
            else:
                st.warning("No recommendations available for this user!")
    
    st.markdown("---")
    
    # ========== HYBRID SEARCH ==========
    st.markdown("""
    <div class="method-card">
        <h2>🟢 Hybrid (CF + NLP) Search</h2>
        <p>Enter book or describe: works with both!</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 1])
    with col1:
        hybrid_query = st.text_input("Search:", 
                                     placeholder="e.g., 'fantasy adventure' or 'Harry Potter'",
                                     key="hybrid_search_input")
    with col2:
        st.write("")
        st.write("")
        search_btn4 = st.button("🔍 Search", key="hybrid_search_btn")
    
    if search_btn4 and hybrid_query:
        with st.spinner("Searching with hybrid approach..."):
            results = recommend_hybrid_search(hybrid_query, data, top_k=5)
            
            if not results.empty:
                st.success(f"Found {len(results)} books!")
                
                # Format kết quả
                display_df = results[['Title', 'Hybrid Score', 'Similarity', 'Impact', 'Avg Rating']].copy()
                display_df['Hybrid Score'] = display_df['Hybrid Score'].apply(lambda x: f"{x:.3f}")
                display_df['Similarity'] = display_df['Similarity'].apply(lambda x: f"{x:.3f}")
                display_df['Impact'] = display_df['Impact'].apply(lambda x: f"{x:.2f}")
                display_df['Avg Rating'] = display_df['Avg Rating'].apply(lambda x: f"{x:.2f}/5")
                
                st.dataframe(display_df, use_container_width=True)
            else:
                st.warning("No results found!")

if __name__ == "__main__":
    main()
import streamlit as st
import nltk
from nltk.corpus import reuters, stopwords
from gensim.models import Word2Vec
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

# Download required NLTK data (this runs only the first time)
nltk.download('reuters')
nltk.download('punkt')
nltk.download('stopwords')

# ---------------------------
# Helper Functions
# ---------------------------
def preprocess_text(text):
    """
    Tokenizes the input text, filters out non-alphanumeric tokens,
    and removes English stop words.
    """
    tokens = nltk.word_tokenize(text)
    tokens = [
        word.lower() for word in tokens 
        if word.isalnum() and word.lower() not in stopwords.words('english')
    ]
    return tokens

def compute_avg_embedding(tokens, model):
    """
    Computes the average Word2Vec embedding for a list of tokens.
    If none of the tokens exist in the model's vocabulary,
    returns a zero vector.
    """
    vecs = [model.wv[word] for word in tokens if word in model.wv]
    if not vecs:
        return np.zeros(model.vector_size)
    return np.mean(vecs, axis=0)

def retrieve_documents(query, model, doc_embeddings, top_n=5):
    """
    Given a query string, computes its average embedding and compares it
    with precomputed document embeddings using cosine similarity.
    Returns the top_n document IDs along with their similarity scores.
    """
    query_tokens = preprocess_text(query)
    query_vec = compute_avg_embedding(query_tokens, model)
    similarities = {}
    for doc_id, doc_vec in doc_embeddings.items():
        # Avoid division by zero if any vector is zero
        if np.linalg.norm(query_vec) == 0 or np.linalg.norm(doc_vec) == 0:
            sim = 0
        else:
            sim = np.dot(query_vec, doc_vec) / (np.linalg.norm(query_vec) * np.linalg.norm(doc_vec))
        similarities[doc_id] = sim
    sorted_docs = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    return sorted_docs[:top_n]

# ---------------------------
# Caching Expensive Operations
# ---------------------------
@st.cache_resource(show_spinner=False)
def load_model_and_doc_embeddings():
    """
    Loads and processes the Reuters corpus:
      - Extracts sentences and trains a Word2Vec model.
      - Precomputes document embeddings for all Reuters documents.
    """
    # Build corpus of tokenized sentences from Reuters documents
    corpus_sentences = []
    for fileid in reuters.fileids():
        raw_text = reuters.raw(fileid)
        tokenized_sentence = [
            word for word in nltk.word_tokenize(raw_text)
            if word.isalnum() and word.lower() not in stopwords.words('english')
        ]
        corpus_sentences.append(tokenized_sentence)

    # Train Word2Vec model
    model = Word2Vec(sentences=corpus_sentences, vector_size=100, window=5, min_count=5, workers=4)

    # Precompute document embeddings
    doc_embeddings = {}
    for doc_id in reuters.fileids():
        raw_text = reuters.raw(doc_id)
        tokens = preprocess_text(raw_text)
        avg_embedding = compute_avg_embedding(tokens, model)
        doc_embeddings[doc_id] = avg_embedding

    return model, doc_embeddings

@st.cache_resource(show_spinner=False)
def compute_tsne_embeddings(_model, num_words=200):
    """
    Computes t-SNE embeddings for the top 'num_words' words in the model vocabulary.
    """
    words = list(_model.wv.index_to_key)[:num_words]
    word_vectors = np.array([_model.wv[word] for word in words])
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    word_vectors_2d = tsne.fit_transform(word_vectors)
    return words, word_vectors_2d

# ---------------------------
# Streamlit App
# ---------------------------
def main():
    st.title("Task 4: Word2Vec Embeddings and Document Retrieval")
    st.sidebar.header("Navigation")
    app_mode = st.sidebar.selectbox("Choose a mode:", ["Visualization", "Document Retrieval"])

    # Load (or train) the model and document embeddings (this is cached)
    with st.spinner("Loading model and data..."):
        model, doc_embeddings = load_model_and_doc_embeddings()

    if app_mode == "Visualization":
        st.header("t-SNE Visualization of Word2Vec Embeddings")
        st.markdown(
            """
            This visualization projects word embeddings into 2D space using t-SNE.
            Use the slider to choose how many of the top words to visualize.
            """
        )
        num_words = st.sidebar.slider("Number of top words to display", 50, 500, 200)
        with st.spinner("Computing t-SNE embeddings..."):
            words, word_vectors_2d = compute_tsne_embeddings(model, num_words=num_words)

        # Plotting using matplotlib
        fig, ax = plt.subplots(figsize=(16, 12))
        ax.scatter(word_vectors_2d[:, 0], word_vectors_2d[:, 1], color='blue', s=20)
        for i, word in enumerate(words):
            ax.text(word_vectors_2d[i, 0] + 0.1, word_vectors_2d[i, 1] + 0.1, word, fontsize=9)
        ax.set_title("Word2Vec Embeddings Visualized with t-SNE")
        ax.set_xlabel("t-SNE Dimension 1")
        ax.set_ylabel("t-SNE Dimension 2")
        st.pyplot(fig)

    elif app_mode == "Document Retrieval":
        st.header("Reuters Document Retrieval System")
        st.markdown(
            """
            Enter a query to retrieve the most relevant Reuters documents using Word2Vec embeddings.
            The system preprocesses the query and compares it to each document's average embedding.
            """
        )
        query = st.text_input("Enter your query:", "oil prices and stock market")
        top_n = st.slider("Number of top documents to retrieve", 1, 10, 5)
        if st.button("Retrieve Documents"):
            with st.spinner("Retrieving documents..."):
                results = retrieve_documents(query, model, doc_embeddings, top_n=top_n)
            if results:
                for doc_id, score in results:
                    st.markdown(f"**Document ID:** {doc_id}  \n**Similarity Score:** {score:.4f}")
                    doc_text = reuters.raw(doc_id)
                    st.text_area("Document Text", doc_text, height=200)
                    st.markdown("---")
            else:
                st.warning("No documents found for the given query.")
    # Enhanced Footer with Icons and Styling
    st.markdown(
    """
    <style>
    .footer {
        text-align: center;
        font-size: 14px;
        color: #555;
        margin-top: 50px;
    }
    .footer a {
        text-decoration: none;
        color: #555;
        margin: 0 10px;
    }
    .footer a:hover {
        color: #000;
    }
    .footer img {
        vertical-align: middle;
        margin-right: 5px;
    }
    </style>
    <hr>
    <div class="footer">
        <p>
            Made with <span style="color: #e25555; font-size: 18px;">&#10084;</span> by <strong>Ahmed Safari - 60101938</strong>
        </p>
        <p>
            <a href="https://github.com/ahmed-safari/Word2VecEmbeddings" target="_blank">
                <img src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png" width="25">
                GitHub Repository
            </a>
        </p>
    </div>
    """,
    unsafe_allow_html=True
)



if __name__ == "__main__":
    main()

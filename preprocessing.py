import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import re
import string
import unicodedata
from pathlib import Path
import streamlit as st
# import nltk
# from nltk.corpus import stopwords
# import torch
# import torch.nn.functional as F
# from nltk.stem import PorterStemmer, WordNetLemmatizer
# from nltk.tokenize import word_tokenize
from multiprocessing import Pool, cpu_count
from functools import partial
from multiprocessing import Pool, cpu_count
import time

# # Download NLTK resources
# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('omw-1.4')


# # Initialize stemmer and lemmatizer (outside the function for efficiency)
# stemmer = PorterStemmer()
# lemmatizer = WordNetLemmatizer()


@st.cache_data
def load_data():
    census_all = pd.read_csv("census_real_estate_data - all places.csv")
    census_county = pd.read_csv("census_real_estate_data_county.csv")
    real_df = pd.read_csv("Connecticut_Louisiana_Real_Estate.csv")
    #social_media_real_estate = pd.read_csv("reddit_real_estate_data_balanced.csv")
    social_media_real_estate = pd.read_csv("processed_sm_real_estate.csv")
    crime_data = pd.read_excel("Crime Data(Louisiana_Connecticut).xlsx")
    return census_all, census_county, real_df, social_media_real_estate, crime_data

census_all, census_county, real_df, social_media_real_estate, crime_data = load_data()
### Cleaning up the Data
data_list = [census_all, census_county, real_df, social_media_real_estate, crime_data]

dataset_names = ["census_all", "census_county", "real_df", "social_media_real_estate", "crime_data"]

# Threshold for missing values
threshold = 0.8

# Function to clean a dataset
def clean_dataset(data, name):
    print(f"Cleaning dataset: {name}")

    # Step 1: Remove columns with more than 80% missing values
    missing_percentage = data.isnull().mean()
    columns_to_drop = missing_percentage[missing_percentage > threshold].index
    print(f"Columns to drop (more than {threshold * 100}% missing values): {list(columns_to_drop)}")
    data.drop(columns=columns_to_drop, inplace=True)

    # Step 2: Handle remaining missing values (fill with median for numeric columns)
    for col in data.columns:
        if data[col].isnull().sum() > 0:
            if data[col].dtype in ['float64', 'int64']:
                data[col].fillna(data[col].median(), inplace=True)
            else:
                data[col].fillna(data[col].mode()[0], inplace=True)

    # Step 3: Remove duplicate rows
    initial_shape = data.shape
    data.drop_duplicates(inplace=True)
    print(f"Removed {initial_shape[0] - data.shape[0]} duplicate rows.")

    # Step 4: Rename "state" column in census_all and census_county to "State_Fips"
    if name in ["census_all", "census_county"]:
        if 'state' in data.columns:
            data.rename(columns={'state': 'State_Fips'}, inplace=True)
            print(f"Renamed 'state' column to 'State_Fips' in {name}.")

    # Step 5: Standardize column names
    data.columns = data.columns.str.lower().str.replace(' ', '_').str.replace('\n', '_')
    print(f"Standardized column names: {list(data.columns)}")

    # Step 6: Display final shape
    print(f"Final shape of {name}: {data.shape}\n")
    return data

# Clean each dataset
for i, data in enumerate(data_list):
    data_list[i] = clean_dataset(data, dataset_names[i])
# for data in data_list:
#     print(data.isnull().sum())

# Initialize preprocessing tools
# stop_words = set(stopwords.words('english'))
# lemmatizer = WordNetLemmatizer()

# @st.cache_resource
# def preprocess_text(text, apply_lemmatization=True):
#     """
#     Enhanced text preprocessing with lemmatization
#     """
#     if not text or not isinstance(text, str):
#         return ""

#     try:
#         # Convert to lowercase
#         text = text.lower()

#         # Remove URLs and HTML tags
#         text = re.sub(r'http\S+|www\S+|https\S+|<.*?>', '', text)

#         # Remove non-ASCII characters but keep emojis
#         text = ''.join([c for c in text if c in string.printable or ord(c) > 127])

#         # Remove special characters
#         text = re.sub(r'[^\w\s.,!?:;()\[\]{}\'"-]', '', text)

#         # Normalize unicode
#         text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')

#         # Tokenize and process words
#         words = word_tokenize(text)
#         words = [word for word in words if word not in stop_words]
        
#         if apply_lemmatization:
#             words = [lemmatizer.lemmatize(word, pos='v') for word in words]  # Verbs
#             words = [lemmatizer.lemmatize(word, pos='n') for word in words]  # Nouns
        
#         return ' '.join(words).strip()
#     except Exception as e:
#         print(f"Error preprocessing text: {e}")
#         return text if isinstance(text, str) else ""

# def parallel_preprocess(series, n_jobs=None):
#     """Parallel text preprocessing"""
#     if n_jobs is None:
#         n_jobs = cpu_count() - 1 if cpu_count() > 1 else 1
    
#     with Pool(n_jobs) as p:
#         results = p.map(preprocess_text, series)
    
#     return pd.Series(results, index=series.index)


# # Preprocess text
# with st.spinner('Preprocessing text...'):
#     social_media_real_estate['content']=social_media_real_estate['content'].fillna("").apply(preprocess_text)
#     social_media_real_estate['comments']=social_media_real_estate['comments'].fillna("").apply(preprocess_text)
#     # social_media_real_estate['content'] = parallel_preprocess(social_media_real_estate['content'].fillna(""))
#     # social_media_real_estate['comments'] = parallel_preprocess(social_media_real_estate['comments'].fillna(""))


# # Display results
# st.success("Preprocessing complete!")
# st.dataframe(social_media_real_estate.head())



# # Example usage inside your app
# # Assuming you already have social_media_real_estate loaded

# # Setup Model
# @st.cache_resource(show_spinner=False)
# def load_sentiment_model():
#     model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     model = AutoModelForSequenceClassification.from_pretrained(model_name, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
#     labels = ['Negative', 'Neutral', 'Positive']
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model = model.to(device)
#     return tokenizer, model, labels, device

# # Prettier Batch Sentiment Analysis
# def get_sentiment_batch_pretty(texts, tokenizer, model, device, labels, batch_size=128):
#     """
#     Analyze Sentiments in Batches with Prettier Streamlit Progress Bar
#     """
#     results = []
#     model.eval()

#     progress_bar = st.progress(0, text="🚀 Starting sentiment analysis...")
#     status_placeholder = st.empty()

#     total_batches = (len(texts) + batch_size - 1) // batch_size

#     for batch_idx, i in enumerate(range(0, len(texts), batch_size)):
#         batch_texts = texts[i:i+batch_size]

#         inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
#         inputs = {k: v.to(device) for k, v in inputs.items()}

#         with torch.no_grad():
#             outputs = model(**inputs)

#         logits = outputs.logits.float()
#         probs = torch.nn.functional.softmax(logits, dim=-1)

#         labels_list = [labels[i] for i in probs.argmax(dim=-1).tolist()]
#         negative_scores = probs[:, 0].tolist()
#         neutral_scores = probs[:, 1].tolist()
#         positive_scores = probs[:, 2].tolist()

#         batch_results = pd.DataFrame({
#             "h_content_sentiment_label": labels_list,
#             "h_content_negative_score": negative_scores,
#             "h_content_neutral_score": neutral_scores,
#             "h_content_positive_score": positive_scores
#         })

#         results.append(batch_results)

#         # Progress Update
#         progress = (batch_idx + 1) / total_batches
#         progress_percent = int(progress * 100)
        
#         progress_bar.progress(progress, text=f"🧠 Analyzing texts... {progress_percent}% complete")
#         status_placeholder.markdown(f"<small>✅ Processed {i+len(batch_texts)}/{len(texts)} texts</small>", unsafe_allow_html=True)

#         time.sleep(0.01)  # Tiny sleep to smooth UI updates

#     progress_bar.empty()
#     status_placeholder.success("🎉 Sentiment analysis complete!")

#     final_results = pd.concat(results, ignore_index=True)
#     return final_results

# # Preprocessing and Sentiment Application Function
# def preprocess_and_sentiment_analysis_pretty(df, text_column):
#     """
#     Takes a DataFrame, cleans it minimally, and adds sentiment analysis columns
#     """
#     tokenizer, model, labels, device = load_sentiment_model()

#     with st.spinner('🛠️ Preparing social media content for analysis...'):
#         content_list = df[text_column].fillna("").astype(str).tolist()
#         sentiment_results = get_sentiment_batch_pretty(content_list, tokenizer, model, device, labels)

#     # Merge back into original dataframe
#     df = df.reset_index(drop=True)
#     df = pd.concat([df, sentiment_results], axis=1)
    
#     return df


# with st.spinner('Analyzing social media content...'):
#     sm_real_estate = preprocess_and_sentiment_analysis_pretty(social_media_real_estate, text_column='content')


# with st.spinner('Analyzing social media comments...'):
#     sm_real_estate = preprocess_and_sentiment_analysis_pretty(sm_real_estate, text_column='comments')

# st.dataframe(sm_real_estate.head())


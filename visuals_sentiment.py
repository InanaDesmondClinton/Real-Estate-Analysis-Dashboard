import streamlit as st
import plotly.express as px
import pandas as pd
from collections import Counter
import nltk
from nltk.util import ngrams
import os
import nltk
from nltk.corpus import stopwords
from collections import Counter
import matplotlib.pyplot as plt




# Ensure punkt is downloaded first
nltk_data_dir = os.path.expanduser('~/nltk_data')
if not os.path.exists(nltk_data_dir + '/tokenizers/punkt'):
    nltk.download('punkt', download_dir=nltk_data_dir)

nltk.data.path.append(nltk_data_dir)  # Tell nltk to look here

@st.cache_data
def plot_sentiment_trends(social_media_real_estate, state):
    state_df = social_media_real_estate[social_media_real_estate['state'] == state]
    content_sentiments = state_df.groupby('city')['content_sentiment_label'].value_counts().unstack(fill_value=0)
    comment_sentiments = state_df.groupby('city')['comments_sentiment_label'].value_counts().unstack(fill_value=0)
    combined_sentiments = content_sentiments.add(comment_sentiments, fill_value=0)
    top_cities = combined_sentiments.sum(axis=1).nlargest(20).index
    top_city_sentiments = combined_sentiments.loc[top_cities]
    st.subheader(f'Sentiment Trends in {state} Cities (Top 20)')
    st.bar_chart(top_city_sentiments)

@st.cache_data
def plot_major_concerns(social_media_real_estate, state):
    state_df = social_media_real_estate[social_media_real_estate['state'] == state]
    concerns = {}
    for _, row in state_df.iterrows():
        text = str(row['content']) + " " + str(row['comments'])
        text = text.lower()
        if "safety" in text or "crime" in text or "dangerous" in text:
            concerns["Safety"] = concerns.get("Safety", 0) + 1
        if "affordability" in text or "expensive" in text or "rent" in text or "mortgage" in text or "cost" in text:
            concerns["Affordability"] = concerns.get("Affordability", 0) + 1
        if "job" in text or "employment" in text or "unemployment" in text or "career" in text or "market" in text:
            concerns["Job Market"] = concerns.get("Job Market", 0) + 1
        if "school" in text or "education" in text:
            concerns["Schools"] = concerns.get("Schools", 0) + 1
        if "infrastructure" in text or "transportation" in text or "roads" in text or "public transit" in text:
            concerns["Infrastructure"] = concerns.get("Infrastructure",0) + 1
        if "taxes" in text or "property tax" in text or "tax rate" in text:
            concerns["Taxes"] = concerns.get("Taxes", 0) + 1
    if concerns:
        st.subheader(f"Major Concerns of Buyers and Renters in {state}")
        st.plotly_chart(px.pie(names=list(concerns.keys()), values=list(concerns.values()), title="Major Concerns"))
    else:
        st.info("No major concerns found for the selected state.")

@st.cache_data
def plot_sentiment_vs_price(real_df, social_media_real_estate, state):
    # Merge the dataframes on 'City'
    real_social = pd.merge(real_df, social_media_real_estate, left_on='city', right_on='city', how='inner')
    real_social = real_social[real_social['state_x'] == state]
    city_sentiment = real_social.groupby('city')['content_sentiment_label'].agg(lambda x: x.value_counts().idxmax())
    real_social = pd.merge(real_social, city_sentiment.rename('major_sentiment'), left_on='city', right_index=True)
    sentiment_mapping = {'Negative': -1, 'Neutral': 0, 'Positive': 1}
    real_social['sentiment_score'] = real_social['major_sentiment'].map(sentiment_mapping)
    st.subheader(f'Correlation between Property Price and Major Sentiment in {state}')
    # Calculate the correlation between property price and sentiment score
    correlation = real_social['median_sale_price'].corr(real_social['sentiment_score'])
    print(f"Correlation between property price and major sentiment: {correlation:.2f}")
    if not real_social.empty:
        fig = px.scatter(real_social, x='sentiment_score', y='median_sale_price',
                         labels={'sentiment_score': 'Sentiment Score', 'median_sale_price': 'Property Price'},
                         title='Correlation between Property Price and Sentiment')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No data available for sentiment vs. price.")

    # Group data by city and sentiment to explore variations
    city_sentiment_price = real_social.groupby(['city', 'major_sentiment'])['median_sale_price'].mean().reset_index()

    # Create an interactive plot to show average price by sentiment for each city
    sentiment_fig = px.scatter(city_sentiment_price, x='major_sentiment', y='median_sale_price', color='city',
                               title='Average Property Price by City and Major Sentiment',
                               labels={'major_sentiment': 'Sentiment', 'median_sale_price': 'Average Property Price'},
                               hover_data=['city', 'median_sale_price'])
    st.plotly_chart(sentiment_fig, use_container_width=True)


# @st.cache_data
# def plot_key_phrases_by_sentiment(social_media_real_estate, state):
#     df = social_media_real_estate[social_media_real_estate['state'] == state].copy()
#     df['final_sentiment'] = df.apply(lambda row: 'Positive' if row['content_sentiment_label'] == 'Positive' or row['comments_sentiment_label'] == 'Positive' else 'Negative', axis=1)
#     def extract_ngrams(text, n=2):
#         tokens = nltk.word_tokenize(text)
#         n_grams = ngrams(tokens, n)
#         return [' '.join(gram) for gram in n_grams]
#     df['content_ngrams'] = df['content'].apply(lambda x: extract_ngrams(x, n=2))
#     df['comment_ngrams'] = df['comments'].apply(lambda x: extract_ngrams(x, n=2))
#     positive_phrases = pd.concat([
#         df[df['final_sentiment'] == 'Positive']['content_ngrams'].explode(),
#         df[df['final_sentiment'] == 'Positive']['comment_ngrams'].explode()
#     ])
#     negative_phrases = pd.concat([
#         df[df['final_sentiment'] == 'Negative']['content_ngrams'].explode(),
#         df[df['final_sentiment'] == 'Negative']['comment_ngrams'].explode()
#     ])
#     pos_top = Counter(positive_phrases).most_common(10)
#     neg_top = Counter(negative_phrases).most_common(10)
#     st.subheader(f'Top Positive and Negative Key Phrases in {state}')
#     st.write('**Positive:**')
#     st.table(pd.DataFrame(pos_top, columns=['Phrase', 'Count']))
#     st.write('**Negative:**')
#     st.table(pd.DataFrame(neg_top, columns=['Phrase', 'Count']))


# Download stopwords (if not already done)
nltk.download('stopwords')
nltk.download('punkt')

# Define stopwords
stop_words = set(stopwords.words('english'))

# Define function to extract n-grams
def extract_ngrams(text, n=2):
    # Tokenize and filter out non-alphabetic and stopwords
    tokens = nltk.word_tokenize(text.lower())  # Make it lowercase for uniformity
    filtered_tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    n_grams = ngrams(filtered_tokens, n)
    return [' '.join(gram) for gram in n_grams]

# Main function to plot key phrases by sentiment
def plot_key_phrases_by_sentiment(social_media_real_estate, state):
    df = social_media_real_estate[social_media_real_estate['state'] == state].copy()
    
    # Determine final sentiment
    df['final_sentiment'] = df.apply(
        lambda row: 'Positive' if row['content_sentiment_label'] == 'Positive' or row['comments_sentiment_label'] == 'Positive' else 'Negative', 
        axis=1
    )
    
    # Extract n-grams for content and comments
    df['content_ngrams'] = df['content'].apply(lambda x: extract_ngrams(x, n=2))
    df['comment_ngrams'] = df['comments'].apply(lambda x: extract_ngrams(x, n=2))
    
    # Extract positive and negative phrases
    positive_phrases = pd.concat([
        df[df['final_sentiment'] == 'Positive']['content_ngrams'].explode(),
        df[df['final_sentiment'] == 'Positive']['comment_ngrams'].explode()
    ])
    
    negative_phrases = pd.concat([
        df[df['final_sentiment'] == 'Negative']['content_ngrams'].explode(),
        df[df['final_sentiment'] == 'Negative']['comment_ngrams'].explode()
    ])
    
    # Get top 10 positive and negative phrases
    pos_top = Counter(positive_phrases).most_common(10)
    neg_top = Counter(negative_phrases).most_common(10)
    
    # Prepare data for Plotly
    pos_df = pd.DataFrame(pos_top, columns=['Phrase', 'Count'])
    neg_df = pd.DataFrame(neg_top, columns=['Phrase', 'Count'])
    
    # Display results using Streamlit
    st.subheader(f'Top Positive and Negative Key Phrases in {state}')
    
    # Plot Top Positive Phrases with Plotly
    st.write('**Positive Phrases:**')
    fig_pos = px.bar(
        pos_df,
        x='Phrase',
        y='Count',
        title=f'Top 20 Positive Key Phrases in {state}',
        labels={'Count': 'Frequency'},
        color='Count',
        color_continuous_scale='Blues',
        template='plotly_dark'
    )
    st.plotly_chart(fig_pos)
    
    # Plot Top Negative Phrases with Plotly
    st.write('**Negative Phrases:**')
    fig_neg = px.bar(
        neg_df,
        x='Phrase',
        y='Count',
        title=f'Top 20 Negative Key Phrases in {state}',
        labels={'Count': 'Frequency'},
        color='Count',
        color_continuous_scale='Reds',
        template='plotly_dark'
    )
    st.plotly_chart(fig_neg)


# Add similar functions for:
# - Sentiment vs. property price (scatter)
# - Key phrases by sentiment 

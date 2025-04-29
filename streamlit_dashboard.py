import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import geopandas as gpd
import folium
from folium.plugins import HeatMap
import nltk
import torch
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import openpyxl
import os 

# Import modular visualization functions
import visuals_louisiana as vis_lou
import visuals_connecticut as vis_ct
import visuals_geospatial as vis_geo
import visuals_sentiment as vis_sent
from preprocessing import census_all, census_county, real_df, social_media_real_estate, crime_data

# Ensure punkt is downloaded first
nltk_data_dir = os.path.expanduser('~/nltk_data')
if not os.path.exists(nltk_data_dir + '/tokenizers/punkt'):
    nltk.download('punkt', download_dir=nltk_data_dir)

nltk.data.path.append(nltk_data_dir)  # Tell nltk to look here

# --- Sidebar Navigation ---
pages = ["General Analysis", "Geospatial Analysis", "Sentiment Analysis"]
page = st.sidebar.radio("Select Page", pages)

# Sidebar filters (common to all pages)
st.sidebar.header("Filters")
states = ["Connecticut", "Louisiana"]
selected_state = st.sidebar.selectbox("Select State", states)
counties = real_df[real_df['state'] == selected_state]['parent_metro_region'].dropna().unique()
selected_county = st.sidebar.selectbox("Select County", np.insert(counties, 0, "All"))
property_types = real_df[real_df['state'] == selected_state]['property_type'].dropna().unique()
selected_property_type = st.sidebar.selectbox("Property Type", np.insert(property_types, 0, "All"))
years = census_all['year'].dropna().unique()
if len(years) > 0:
    min_year, max_year = int(np.min(years)), int(np.max(years))
    selected_year_range = st.sidebar.slider("Year Range for Property Trends", min_year, max_year, (min_year, max_year))
else:
    selected_year_range = (None, None)
crime_rate_bins = ["All", "Low", "Medium", "High"]
selected_crime_rate = st.sidebar.selectbox("Crime Rate", crime_rate_bins)
sentiment_score_bins = ["All", "Positive", "Neutral", "Negative"]
selected_sentiment = st.sidebar.selectbox("Sentiment Score", sentiment_score_bins)

crime_data['total_crimes'] = crime_data['violent_crime'] + crime_data['property_crime'] + crime_data['arson3']
crime_data['crime_rate'] = (crime_data['total_crimes'] / crime_data['population']) * 1000

# %%
# Clean city names
census_all = census_all[~census_all["name"].str.contains("CDP|town|village", case=False, na=False)]
census_all['base_city'] = census_all['name'].str.extract(r'^([\w\s\-]+) (city|town|village),')[0]

st.title("Real Estate Analysis Dashboard")
st.write("Use the sidebar to navigate between analysis pages and filter the data.")

# --- General Analysis Page ---
if page == "General Analysis":
    st.header("General Real Estate & Crime Analysis")
    general_visuals = [
        "Average Housing Prices by County",
        "Crime Rate vs. Median Sale Price",
        "Top 5 Safest/Dangerous Counties",
        "Household Income vs. Real Estate Price",
        "Property Type Distribution",
        "Rental vs. Owner-Occupied Distribution",
        "Economic Indicators Correlation"
    ]
    # Session state for shown visuals
    if 'general_shown' not in st.session_state:
        st.session_state.general_shown = [True] + [False]*(len(general_visuals)-1)
    # Show the first visual
    if selected_state == "Louisiana":
        vis_lou.plot_avg_housing_prices(real_df, selected_state, selected_county, selected_property_type, selected_year_range)
    else:
        vis_ct.plot_avg_housing_prices(real_df, selected_state, selected_county, selected_property_type, selected_year_range)
    # Show buttons for other visuals
    for i, visual in enumerate(general_visuals[1:], start=1):
        if not st.session_state.general_shown[i]:
            if st.button(f"Show {visual}"):
                st.session_state.general_shown[i] = True
        if st.session_state.general_shown[i]:
            if visual == general_visuals[1]:
                if selected_state == "Louisiana":
                    vis_lou.plot_crime_vs_price(real_df, crime_data, selected_state, selected_county, selected_property_type, selected_year_range)
                else:
                    vis_ct.plot_crime_vs_price(real_df, crime_data, selected_state, selected_county, selected_property_type, selected_year_range)
            elif visual == general_visuals[2]:
                if selected_state == "Louisiana":
                    vis_lou.plot_top5_safest_dangerous(real_df, crime_data, selected_state, selected_county, selected_property_type, selected_year_range)
                else:
                    vis_ct.plot_top5_safest_dangerous(real_df, crime_data, selected_state, selected_county, selected_property_type, selected_year_range)
            elif visual == general_visuals[3]:
                if selected_state == "Louisiana":
                    vis_lou.plot_income_vs_price(real_df, census_all, selected_state, selected_county, selected_property_type, selected_year_range)
                else:
                    vis_ct.plot_income_vs_price(real_df, census_all, selected_state, selected_county, selected_property_type, selected_year_range)
            elif visual == general_visuals[4]:
                if selected_state == "Louisiana":
                    vis_lou.plot_property_type_distribution(real_df, selected_state, selected_county, selected_property_type, selected_year_range)
                else:
                    vis_ct.plot_property_type_distribution(real_df, selected_state, selected_county, selected_property_type, selected_year_range)
            elif visual == general_visuals[5]:
                if selected_state == "Louisiana":
                    real_census = vis_lou.prepare_real_census_per_state(real_df, census_all, selected_state)  # Cached after first run
                    vis_lou.plot_rental_vs_owner(real_census, selected_state, selected_county, selected_property_type, selected_year_range)  # Instant
                    #vis_lou.plot_rental_vs_owner(real_df, census_all, selected_state, selected_county, selected_property_type, selected_year_range)
                else:
                    real_census = vis_ct.prepare_real_census_per_state(real_df, census_all, selected_state)
                    vis_ct.plot_rental_vs_owner(real_census, selected_state, selected_county, selected_property_type, selected_year_range)
            elif visual == general_visuals[6]:
                if selected_state == "Louisiana":
                    real_census = vis_lou.prepare_real_census_per_state(real_df, census_all, selected_state)
                    vis_lou.plot_economic_correlation(real_census, selected_state, selected_county, selected_property_type, selected_year_range)
                else:
                    real_census = vis_ct.prepare_real_census_per_state(real_df, census_all, selected_state)
                    vis_ct.plot_economic_correlation(real_census, selected_state, selected_county, selected_property_type, selected_year_range)

# --- Geospatial Analysis Page ---
elif page == "Geospatial Analysis":
    st.header("Geospatial Analysis")
    # Immediately show the Crime Rate Analysis
    #vis_geo.crime_rate_analysis(census_all, crime_data, selected_state)
    geo_visuals = [
        "Crime Rate Analysis",
        "Crime Rate Choropleth",
        "Property Value Choropleth",
        "Population Change Analysis",
        "Gentrification Analysis",
        "Gentrification Heatmap"
    ]
    if 'geo_shown' not in st.session_state:
        st.session_state.geo_shown = [True] + [False]*(len(geo_visuals)-1)
    #geojson_url = 'https://raw.githubusercontent.com/PublicaMundi/MappingAPI/master/data/geojson/us-states.json'
    vis_geo.crime_rate_analysis(census_all, crime_data, selected_state)
    for i, visual in enumerate(geo_visuals[1:], start=1):   # <-- No start=1:
        if not st.session_state.geo_shown[i]:
            if st.button(f"Show {visual}"):
                st.session_state.geo_shown[i] = True
        if st.session_state.geo_shown[i]:
            # if visual == geo_visuals[0]:
            #     vis_geo.crime_rate_analysis(census_all, crime_data, selected_state)
            if visual == geo_visuals[1]:
                vis_geo.plot_crime_rate_choropleth(crime_data)
            if visual == geo_visuals[2]:
                vis_geo.plot_property_value_choropleth(census_all)
            elif visual == geo_visuals[3]:
                vis_geo.plot_population_change(census_all, selected_state)
            elif visual == geo_visuals[4]:
                vis_geo.gentrification_analysis(census_all, selected_state)
            elif visual == geo_visuals[5]:
                vis_geo.choropleth_heat_map(census_county)


# --- Sentiment Analysis Page ---
elif page == "Sentiment Analysis":
    st.header("Sentiment Analysis")
    sentiment_visuals = [
        "Sentiment Trends by City",
        "Major Concerns",
        "Sentiment vs. Property Price",
        "Key Phrases by Sentiment"
    ]
    if 'sentiment_shown' not in st.session_state:
        st.session_state.sentiment_shown = [True] + [False]*(len(sentiment_visuals)-1)
    vis_sent.plot_sentiment_trends(social_media_real_estate, selected_state)
    for i, visual in enumerate(sentiment_visuals[1:], start=1):
        if not st.session_state.sentiment_shown[i]:
            if st.button(f"Show {visual}"):
                st.session_state.sentiment_shown[i] = True
        if st.session_state.sentiment_shown[i]:
            if visual == sentiment_visuals[1]:
                vis_sent.plot_major_concerns(social_media_real_estate, selected_state)
            elif visual == sentiment_visuals[2]:
                vis_sent.plot_sentiment_vs_price(real_df, social_media_real_estate, selected_state)
            elif visual == sentiment_visuals[3]:
                vis_sent.plot_key_phrases_by_sentiment(social_media_real_estate, selected_state) 
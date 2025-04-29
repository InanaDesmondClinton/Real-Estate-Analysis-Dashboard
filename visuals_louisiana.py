import streamlit as st
import plotly.express as px
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

@st.cache_data
def plot_avg_housing_prices(real_df, state, county, property_type, year_range):
    df = real_df[real_df['state'] == state]
    if county != "All":
        df = df[df['parent_metro_region'] == county]
    if property_type != "All":
        df = df[df['property_type'] == property_type]
    # if year_range[0] is not None and year_range[1] is not None:
    #     df = df[(df['year'] >= year_range[0]) & (df['year'] <= year_range[1])]
    st.subheader(f"Average Housing Prices by County in {state}")
    if not df.empty:
        avg_price = df.groupby('parent_metro_region')['median_sale_price'].mean().reset_index()
        avg_price = avg_price.sort_values('median_sale_price', ascending=False)
        fig = px.bar(avg_price, x='parent_metro_region', y='median_sale_price',
                    labels={'parent_metro_region': 'County', 'median_sale_price': 'Average Sale Price ($)'},
                    color='median_sale_price', color_continuous_scale='viridis', height=500)
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No data available for the selected filters.")

@st.cache_data
def plot_crime_vs_price(real_df, crime_data, state, county, property_type, year_range):
    crime_merge = pd.merge(
        real_df,
        crime_data,
        left_on=['city', 'state'],
        right_on=['city', 'state'],
        how='inner'
    )
    crime_merge = crime_merge[crime_merge['state'] == state]
    if county != "All":
        crime_merge = crime_merge[crime_merge['parent_metro_region'] == county]
    if property_type != "All":
        crime_merge = crime_merge[crime_merge['property_type'] == property_type]
    if year_range[0] is not None and year_range[1] is not None and 'year' in crime_merge.columns:
        crime_merge = crime_merge[(crime_merge['year'] >= year_range[0]) & (crime_merge['year'] <= year_range[1])]
    if 'total_crimes' not in crime_merge.columns:
        crime_merge['total_crimes'] = crime_merge[['violent_crime', 'property_crime', 'arson3']].sum(axis=1)
    if 'crime_rate' not in crime_merge.columns:
        crime_merge['crime_rate'] = crime_merge['total_crimes'] / crime_merge['population']
    st.subheader(f"Crime Rate vs. Median Sale Price by County in {state}")
    if not crime_merge.empty:
        fig2 = px.scatter(
            crime_merge,
            x='crime_rate',
            y='median_sale_price',
            color='parent_metro_region',
            hover_data=['city', 'parent_metro_region', 'median_sale_price', 'crime_rate'],
            labels={'crime_rate': 'Crime Rate', 'median_sale_price': 'Median Sale Price ($)'},
            title=f'Crime Rate vs. Median Sale Price by County in {state}',
            height=500
        )
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("No data available for the selected filters.")

@st.cache_data
def plot_top5_safest_dangerous(real_df, crime_data, state, county, property_type, year_range):
    crime_merge = pd.merge(
        real_df,
        crime_data,
        left_on=['city', 'state'],
        right_on=['city', 'state'],
        how='inner'
    )
    crime_merge = crime_merge[crime_merge['state'] == state]
    if county != "All":
        crime_merge = crime_merge[crime_merge['parent_metro_region'] == county]
    if property_type != "All":
        crime_merge = crime_merge[crime_merge['property_type'] == property_type]
    if year_range[0] is not None and year_range[1] is not None and 'year' in crime_merge.columns:
        crime_merge = crime_merge[(crime_merge['year'] >= year_range[0]) & (crime_merge['year'] <= year_range[1])]
    if 'total_crimes' not in crime_merge.columns:
        crime_merge['total_crimes'] = crime_merge[['violent_crime', 'property_crime', 'arson3']].sum(axis=1)
    if 'crime_rate' not in crime_merge.columns:
        crime_merge['crime_rate'] = crime_merge['total_crimes'] / crime_merge['population']
    if not crime_merge.empty:
        county_rates = crime_merge.groupby('parent_metro_region', as_index=False)['crime_rate'].mean()
        top5_safest = county_rates.sort_values('crime_rate', ascending=True).head(5)
        top5_dangerous = county_rates.sort_values('crime_rate', ascending=False).head(5)
        st.subheader(f"Top 5 Safest Counties in {state} by Crime Rate")
        fig3 = px.bar(top5_safest, x='parent_metro_region', y='crime_rate', color='crime_rate', color_continuous_scale='viridis',
                    labels={'parent_metro_region': 'County', 'crime_rate': 'Crime Rate'}, height=400)
        fig3.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig3, use_container_width=True)
        st.subheader(f"Top 5 Most Dangerous Counties in {state} by Crime Rate")
        fig4 = px.bar(top5_dangerous, x='parent_metro_region', y='crime_rate', color='crime_rate', color_continuous_scale='viridis',
                    labels={'parent_metro_region': 'County', 'crime_rate': 'Crime Rate'}, height=400)
        fig4.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig4, use_container_width=True)
    else:
        st.info("No data available for the selected filters.")

@st.cache_data
def plot_income_vs_price(real_df, census_all, state, county, property_type, year_range):
    census_all = census_all.copy()
    census_all = census_all[~census_all["name"].str.contains("CDP|town|village", case=False, na=False)]
    census_all.rename(columns={'state': 'census_state'}, inplace=True)
    census_all['base_city'] = census_all['name'].str.extract(r'^([\w\s\-]+) (city|town|village),')[0]
    real_census = pd.merge(real_df, census_all, left_on='city', right_on='base_city', how='inner')
    df = real_census[real_census["state"]==state]
    if county != "All":
        df = df[df['parent_metro_region'] == county]
    if property_type != "All":
        df = df[df['property_type'] == property_type]
    if year_range[0] is not None and year_range[1] is not None and 'year' in df.columns:
        df = df[(df['year'] >= year_range[0]) & (df['year'] <= year_range[1])]
    df['b19013_001e'] = pd.to_numeric(df['b19013_001e'], errors='coerce')
    df = df.dropna(subset=['b19013_001e', 'median_sale_price', 'parent_metro_region'])
    st.subheader(f"Household Income vs. Real Estate Price by County in {state}")
    if not df.empty:
        fig = px.scatter(df, x='b19013_001e', y='median_sale_price', color='parent_metro_region',
                         labels={'b19013_001e': 'Household Income', 'median_sale_price': 'Median Sale Price'},
                         hover_data=['parent_metro_region', 'city'], height=500)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No data available for the selected filters.")

# @st.cache_data
# def plot_rental_vs_owner(real_census, state, county, property_type, year_range):
#     # Filter data based on the state first
#     df = real_census[real_census["state"] == state]
    
#     # Apply county filter if not "All"
#     if county != "All":
#         df = df[df['parent_metro_region'] == county]
    
#     # Apply property type filter if not "All"
#     if property_type != "All":
#         df = df[df['property_type'] == property_type]
    
#     # Apply year filter if year range is provided
#     if year_range[0] is not None and year_range[1] is not None and 'year' in df.columns:
#         df = df[(df['year'] >= year_range[0]) & (df['year'] <= year_range[1])]
    
#     # Calculate total units and percentages in one go
#     df['total_units'] = df['b25003_002e'] + df['b25003_003e']
#     df['b25003_002e_percentage'] = (df['b25003_002e'] / df['total_units']) * 100
#     df['b25003_003e_percentage'] = (df['b25003_003e'] / df['total_units']) * 100
    
#     # Only sort if necessary and on a smaller subset
#     df = df.sort_values(by='b25003_002e_percentage', ascending=False)

#     st.subheader(f"Distribution of Rental vs Owner-Occupied Properties by County in {state}")
    
#     # Only plot if data is available after filtering
#     if not df.empty:
#         # Plot using Plotly bar chart
#         fig = px.bar(df, x='parent_metro_region', y=['b25003_002e_percentage', 'b25003_003e_percentage'],
#                      labels={'value': 'Percentage of Total Housing Units', 'parent_metro_region': 'County', 'variable': 'Type'},
#                      barmode='group', height=500)
#         st.plotly_chart(fig, use_container_width=True)
#     else:
#         st.info("No data available for the selected filters.")

@st.cache_data
def plot_rental_vs_owner(real_census, state, county, property_type, year_range):
    # Filter once based on all conditions
    df = real_census[
        (real_census["state"] == state) &
        (real_census['parent_metro_region'] == county if county != "All" else real_census['parent_metro_region']) &
        (real_census['property_type'] == property_type if property_type != "All" else real_census['property_type']) &
        ((real_census['year'] >= year_range[0]) & (real_census['year'] <= year_range[1]) if None not in year_range else True)
    ]
    
    # Check if the filtered DataFrame is empty
    if df.empty:
        st.info("No data available for the selected filters.")
        return
    
    # Calculate percentages after filtering
    df['total_units'] = df['b25003_002e'] + df['b25003_003e']
    df['Owner_percentage'] = df['b25003_002e'] / df['total_units'] * 100
    df['Rental_percentage'] = df['b25003_003e'] / df['total_units'] * 100
    df = df.sort_values(by='Owner_percentage', ascending=False)

    # Plotting
    st.subheader(f"Distribution of Rental vs Owner-Occupied Properties by County in {state}")
    fig = px.bar(df, x='parent_metro_region', y=['Owner_percentage', 'Rental_percentage'],
                 labels={'value': 'Percentage of Total Housing Units', 'parent_metro_region': 'County', 'variable': 'Type'},
                 barmode='group', height=500)
    st.plotly_chart(fig, use_container_width=True)


@st.cache_data
def plot_property_type_distribution(real_df, state, county, property_type, year_range):
    df = real_df[real_df['state'] == state]
    if county != "All":
        df = df[df['parent_metro_region'] == county]
    if property_type != "All":
        df = df[df['property_type'] == property_type]
    # if year_range[0] is not None and year_range[1] is not None:
    #     df = df[(df['year'] >= year_range[0]) & (df['year'] <= year_range[1])]
    st.subheader(f"Property Types Distribution in {state}")
    if not df.empty:
        property_counts = df['property_type'].value_counts()
        fig = px.pie(values=property_counts.values, names=property_counts.index, title='Property Types Distribution')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No data available for the selected filters.")

# @st.cache_data
# def prepare_real_census(real_df, census_all):
#     census_all = census_all.copy()
#     census_all = census_all[~census_all["name"].str.contains("CDP|town|village", case=False, na=False)]
#     census_all.rename(columns={'state': 'census_state'}, inplace=True)
#     census_all['base_city'] = census_all['name'].str.extract(r'^([\w\s\-]+) (city|town|village),')[0]
#     real_census = pd.merge(real_df, census_all, left_on='city', right_on='base_city', how='inner')
#     return real_census

@st.cache_data
def prepare_real_census_per_state(real_df, census_all, state):
    census_all = census_all.copy()
    census_all = census_all[~census_all["name"].str.contains("CDP|town|village", case=False, na=False)]
    census_all.rename(columns={'state': 'census_state'}, inplace=True)
    census_all['base_city'] = census_all['name'].str.extract(r'^([\w\s\-]+) (city|town|village),')[0]
    
    real_df_filtered = real_df[real_df['state'] == state]
    census_all_filtered = census_all[census_all['census_state'] == state]
    
    real_census = pd.merge(real_df_filtered, census_all_filtered, left_on='city', right_on='base_city', how='inner')
    return real_census

@st.cache_data
def plot_rental_vs_owner(real_census, state, county, property_type, year_range):
    df = real_census[real_census["state"]==state]
    if county != "All":
        df = df[df['parent_metro_region'] == county]
    if property_type != "All":
        df = df[df['property_type'] == property_type]
    if year_range[0] is not None and year_range[1] is not None and 'year' in df.columns:
        df = df[(df['year'] >= year_range[0]) & (df['year'] <= year_range[1])]
    df['total_units'] = df['b25003_002e'] + df['b25003_003e']
    df['b25003_002e_percentage'] = df['b25003_002e'] / df['total_units'] * 100
    df['b25003_003e_percentage'] = df['b25003_003e'] / df['total_units'] * 100
    df = df.sort_values(by='b25003_002e_percentage', ascending=False)

    st.subheader(f"Distribution of Rental vs Owner-Occupied Properties by County in {state}")
    if not df.empty:
        fig = px.bar(df, x='parent_metro_region', y=['b25003_002e_percentage', 'b25003_003e_percentage'],
                     labels={'value': 'Percentage of Total Housing Units', 'parent_metro_region': 'County', 'variable': 'Type'},
                     barmode='group', height=500)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No data available for the selected filters.")

@st.cache_data
def plot_economic_correlation(real_census, state, county, property_type, year_range):
    df = real_census[real_census["state"]==state]
    if county != "All":
        df = df[df['parent_metro_region'] == county]
    if property_type != "All":
        df = df[df['property_type'] == property_type]
    if year_range[0] is not None and year_range[1] is not None and 'year' in df.columns:
        df = df[(df['year'] >= year_range[0]) & (df['year'] <= year_range[1])]
    df = df.rename(columns={'b19013_001e': 'median_household_income', 
                             'b19301_001e': 'per_capita_income', 
                             'b23025_005e': 'unemployment_rate', 
                             'b17001_002e': 'poverty_rate', 
                             'b25077_001e': 'median_home_value', 
                             'b25064_001e': 'median_gross_rent', 
                             'b25003_002e': 'homeownership_rate', 
                             'b25002_002e': 'vacancy_rate'})
    correlation_data = df[['median_household_income',
                          'per_capita_income', 
                          'unemployment_rate', 
                          'poverty_rate', 
                          'median_home_value', 
                          'median_gross_rent', 
                          'homeownership_rate', 
                          'vacancy_rate', 
                          'median_sale_price']]
    correlation_matrix = correlation_data.corr()
    st.subheader(f"Correlation between Economic Indicators and Real Estate Activity in {state}")
    if not correlation_matrix.empty:
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax)
        st.pyplot(fig)
    else:
        st.info("No data available for the selected filters.")

# Add similar functions for:
# - Household Income vs. Real Estate Price
# - Property Type Distribution
# - Rental vs. Owner-Occupied Distribution
# - Economic Indicators Correlation 
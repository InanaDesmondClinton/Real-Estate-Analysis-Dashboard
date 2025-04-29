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

@st.cache_data(ttl=3600)  # Cache for 1 hour
def plot_crime_vs_price(real_df, crime_data, state, county, property_type, year_range):
    """
    Optimized function to plot crime rate vs median sale price
    """
    # Input validation
    if not isinstance(year_range, (list, tuple)) or len(year_range) != 2:
        st.error("Invalid year range format")
        return
    
    with st.spinner("Processing crime and price data..."):
        # Filter the datasets first before merging (much more efficient)
        real_filtered = real_df[real_df['state'] == state].copy()
        crime_filtered = crime_data[crime_data['state'] == state].copy()
        
        # Early return if no data for state
        if real_filtered.empty or crime_filtered.empty:
            st.info(f"No data available for state: {state}")
            return
        
        # Apply additional filters before merging
        if county != "All":
            real_filtered = real_filtered[real_filtered['parent_metro_region'] == county].copy()
            crime_filtered = crime_filtered[crime_filtered['parent_metro_region'] == county].copy()
        
        if property_type != "All":
            real_filtered = real_filtered[real_filtered['property_type'] == property_type].copy()
        
        # Merge the pre-filtered datasets
        crime_merge = pd.merge(
            real_filtered,
            crime_filtered,
            on=['city', 'state'],
            how='inner'  # Only keep matching records
        )
        
        # Apply year filter if needed
        if year_range[0] is not None and year_range[1] is not None and 'year' in crime_merge.columns:
            crime_merge = crime_merge[
                (crime_merge['year'] >= year_range[0]) & 
                (crime_merge['year'] <= year_range[1])
            ].copy()
        
        # Early return if no data after filtering
        if crime_merge.empty:
            st.info("No data available for the selected filters.")
            return
        
        # Calculate crime metrics
        crime_cols = ['violent_crime', 'property_crime', 'arson3']
        crime_merge['total_crimes'] = crime_merge[crime_cols].sum(axis=1)
        crime_merge['crime_rate'] = crime_merge['total_crimes'] / crime_merge['population']
        
        # Sample if dataset is too large
        if len(crime_merge) > 5000:  # Reduced from 10,000 for scatter plots
            crime_merge = crime_merge.sample(5000, random_state=42)
            st.warning("Displaying a sample of 5,000 records for better performance")
    
    # Plotting section
    st.subheader(f"Crime Rate vs. Median Sale Price by County in {state}")
    
    if not crime_merge.empty:
        # Create figure with optimized parameters
        fig = px.scatter(
            crime_merge,
            x='crime_rate',
            y='median_sale_price',
            color='parent_metro_region',
            hover_data={
                'city': True,
                'parent_metro_region': True,
                'median_sale_price': ':.2f',
                'crime_rate': ':.4f',
                'total_crimes': True
            },
            labels={
                'crime_rate': 'Crime Rate (per capita)',
                'median_sale_price': 'Median Sale Price ($)',
                'parent_metro_region': 'County'
            },
            height=600,  # Slightly taller for better visibility
            render_mode='webgl',  # Faster rendering
            trendline='lowess',  # Optional: show trend line
            trendline_options=dict(frac=0.3)  # Smoothing factor
        )
        
        # Improve layout
        fig.update_layout(
            hovermode='closest',
            xaxis_title='Crime Rate (crimes per capita)',
            yaxis_title='Median Sale Price ($)',
            legend_title='Counties'
        )
        
        st.plotly_chart(fig, use_container_width=True)

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

@st.cache_data(ttl=3600)  # Cache for 1 hour
def plot_rental_vs_owner(real_census, state, county, property_type, year_range):
    """
    Optimized function to plot rental vs owner-occupied properties distribution
    """
    # Input validation
    if not isinstance(year_range, (list, tuple)) or len(year_range) != 2:
        st.error("Invalid year range format")
        return
    
    with st.spinner("Processing data..."):
        # Create initial filtered copy
        df = real_census[real_census["state"] == state].copy()
        
        # Early return if no state data
        if df.empty:
            st.info(f"No data available for state: {state}")
            return
        
        # Apply all filters
        if county != "All":
            df = df[df['parent_metro_region'] == county].copy()
        
        if property_type != "All":
            df = df[df['property_type'] == property_type].copy()
        
        if year_range[0] is not None and year_range[1] is not None and 'year' in df.columns:
            df = df[(df['year'] >= year_range[0]) & (df['year'] <= year_range[1])].copy()
        
        # Sample if dataset is too large
        if len(df) > 10000:
            df = df.sample(10000, random_state=42)
            st.warning("Displaying a sample of 10,000 records for performance")
        
        # Early return if no data after filtering
        if df.empty:
            st.info("No data available for the selected filters.")
            return
        
        # Calculate percentages
        total_units = df['b25003_002e'] + df['b25003_003e']
        owner_pct = (df['b25003_002e'] / total_units) * 100
        rental_pct = (df['b25003_003e'] / total_units) * 100
        
        # Create plotting dataframe
        plot_df = pd.DataFrame({
            'County': df['parent_metro_region'],
            'Owner_percentage': owner_pct,
            'Rental_percentage': rental_pct
        }).dropna()
        
        # Sort by owner percentage
        plot_df = plot_df.sort_values('Owner_percentage', ascending=False)
    
    # Plotting section
    st.subheader(f"Distribution of Rental vs Owner-Occupied Properties by County in {state}")
    
    if not plot_df.empty:
        # Melt for efficient plotting
        melted_df = plot_df.melt(
            id_vars=['County'], 
            value_vars=['Owner_percentage', 'Rental_percentage'],
            var_name='Type', 
            value_name='Percentage'
        )
        
        fig = px.bar(
            melted_df, 
            x='County', 
            y='Percentage', 
            color='Type',
            labels={'Percentage': 'Percentage of Total Housing Units'},
            barmode='group', 
            height=500
        )
        
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

# @st.cache_data
# def plot_rental_vs_owner(real_census, state, county, property_type, year_range):
#     df = real_census[real_census["state"]==state]
#     if county != "All":
#         df = df[df['parent_metro_region'] == county]
#     if property_type != "All":
#         df = df[df['property_type'] == property_type]
#     if year_range[0] is not None and year_range[1] is not None and 'year' in df.columns:
#         df = df[(df['year'] >= year_range[0]) & (df['year'] <= year_range[1])]
#     df['total_units'] = df['b25003_002e'] + df['b25003_003e']
#     df['Owner_percentage'] = df['b25003_002e'] / df['total_units'] * 100
#     df['Rental_percentage'] = df['b25003_003e'] / df['total_units'] * 100
#     df = df.sort_values(by='Owner_percentage', ascending=False)

#     st.subheader(f"Distribution of Rental vs Owner-Occupied Properties by County in {state}")
#     if not df.empty:
#         fig = px.bar(df, x='parent_metro_region', y=['Owner_percentage', 'Rental_percentage'],
#                      labels={'value': 'Percentage of Total Housing Units', 'parent_metro_region': 'County', 'variable': 'Type'},
#                      barmode='group', height=500)
#         st.plotly_chart(fig, use_container_width=True)
#     else:
#         st.info("No data available for the selected filters.")

@st.cache_data
def plot_rental_vs_owner(real_census, state, county, property_type, year_range):
    # Filter data based on the state first
    df = real_census[real_census["state"] == state]
    
    # Apply county filter if not "All"
    if county != "All":
        df = df[df['parent_metro_region'] == county]
    
    # Apply property type filter if not "All"
    if property_type != "All":
        df = df[df['property_type'] == property_type]
    
    # Apply year filter if year range is provided
    if year_range[0] is not None and year_range[1] is not None and 'year' in df.columns:
        df = df[(df['year'] >= year_range[0]) & (df['year'] <= year_range[1])]
    
    # Calculate total units and percentages in one go
    df['total_units'] = df['b25003_002e'] + df['b25003_003e']
    df['Owner_percentage'] = (df['b25003_002e'] / df['total_units']) * 100
    df['Rental_percentage'] = (df['b25003_003e'] / df['total_units']) * 100
    
    # Only sort if necessary and on a smaller subset
    df = df.sort_values(by='Owner_percentage', ascending=False)

    st.subheader(f"Distribution of Rental vs Owner-Occupied Properties by County in {state}")
    
    # Only plot if data is available after filtering
    if not df.empty:
        # Plot using Plotly bar chart
        fig = px.bar(df, x='parent_metro_region', y=['Owner_percentage', 'Rental_percentage'],
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

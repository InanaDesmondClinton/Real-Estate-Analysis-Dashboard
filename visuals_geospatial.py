import streamlit as st
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots



@st.cache_data
def crime_rate_analysis(census_all, crime_data, state):
    if crime_data is None:
        print("Crime data not available for analysis")
        return

    # Filter by state
    census_all = census_all[census_all["state"] == state]
    crime_data = crime_data[crime_data["state"] == state]

    # Merge datasets
    merged_df = pd.merge(census_all, crime_data, left_on='base_city', right_on='city', how='inner')

    if merged_df.empty:
        print(f"No matching data found for state: {state}")
        return

    # Calculate crime rate per 1000 people
    merged_df['total_crimes'] = merged_df['violent_crime'] + merged_df['property_crime'] + merged_df['arson3']
    merged_df['crime_rate'] = (merged_df['total_crimes'] / merged_df['b01003_001e']) * 1000

    # Plot 1: Crime rate by city
    fig1 = px.bar(
        merged_df.sort_values('crime_rate', ascending=False),
        x='name',
        y='crime_rate',
        color='state_x',
        title=f'Crime Rate per 1000 People by City - {state}',
        labels={'name': 'City', 'crime_rate': 'Crime Rate per 1000 People'}
    )

    fig1.update_layout(
        xaxis_tickangle=-90,
        xaxis_title='City',
        yaxis_title='Crime Rate per 1000 People',
        title_x=0.5
    )
    st.plotly_chart(fig1, use_container_width=True)

    # Plot 2: Crime rate vs. median home value
    fig2 = px.scatter(
        merged_df,
        x='crime_rate',
        y='b25077_001e',
        color='state_x',
        size='b01003_001e',
        size_max=50,
        opacity=0.7,
        title=f'Crime Rate vs. Median Home Value - {state}',
        labels={'crime_rate': 'Crime Rate per 1000 People', 'b25077_001e': 'Median Home Value ($)'},
        hover_data=['name']
    )

    fig2.update_layout(
        xaxis_title='Crime Rate per 1000 People',
        yaxis_title='Median Home Value ($)',
        title_x=0.5
    )
    st.plotly_chart(fig2, use_container_width=True)

@st.cache_data
def plot_crime_rate_choropleth(crime_data):
    with open('us-states.json', 'r') as f:
        geojson_data = json.load(f)
    st.subheader("Crime Rate per 1000 People by State")
    # st.write(crime_data['state'].unique())
    # st.write([feature['properties']['name'] for feature in geojson_data['features']])
    if 'state' in crime_data.columns and 'crime_rate' in crime_data.columns:
        fig = px.choropleth_mapbox(
            crime_data,
            geojson=geojson_data,
            locations='state',
            featureidkey='properties.name',
            color='crime_rate',
            hover_name='state',
            color_continuous_scale='YlOrRd',
            mapbox_style='carto-positron',
            zoom=3,
            center={"lat": 37.0902, "lon": -95.7129},
            title='Crime Rate per 1000 People by State'
        )
        fig.update_layout(margin={"r":0,"t":30,"l":0,"b":0})
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No data available for choropleth.")

@st.cache_data
def plot_property_value_choropleth(census_all):
    with open('us-states.json', 'r') as f:
        geojson_data = json.load(f)
    st.subheader("Median Property Value by State (2023)")
    recent_data = census_all[census_all['year'] == 2023].copy()
    bins = [0, 100000, 150000, 200000, 300000, 500000]
    labels = ['Under $100K', '$100K-$150K', '$150K-$200K', '$200K-$300K', 'Over $300K']
    recent_data['value_category'] = pd.cut(recent_data['b25077_001e'], bins=bins, labels=labels)
    # Optional: Bar chart distribution
    value_counts = pd.crosstab(recent_data['state'], recent_data['value_category'])
    # Convert crosstab to a proper dataframe for Plotly
    value_counts = value_counts.reset_index().melt(id_vars='state', var_name='Value Category', value_name='Count')

    # Now plot with Plotly Express
    fig = px.bar(
        value_counts,
        x='state',
        y='Count',
        color='Value Category',
        title='Stacked Bar Chart of Value Category by State',
        text_auto=True,   # shows count numbers automatically
    )

    fig.update_layout(
        barmode='stack',         # Stacked bars
        xaxis_title='State',
        yaxis_title='Count',
        title_x=0.5,             # Center the title
        coloraxis_colorbar_title='Value Category',
        #template='plotly_viridis',  # similar to 'viridis' colormap
    )

    st.plotly_chart(fig, use_container_width=True)

    if not recent_data.empty:
        fig = px.choropleth_mapbox(
            recent_data,
            geojson=geojson_data,
            locations='state',
            featureidkey='properties.name',
            color='value_category',
            mapbox_style='carto-positron',
            zoom=3,
            center={"lat": 37.0902, "lon": -95.7129},
            hover_name='state',
            hover_data={'b25077_001e': True},
            title='Median Property Value by State (2023)'
        )
        fig.update_layout(margin={"r":0,"t":30,"l":0,"b":0})
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No data available for property value choropleth.")

@st.cache_data
def plot_population_change(census_all, state):
    st.subheader(f'Population vs. Home Value Change in {state} (2015–2017)')
    
    # Filter data for the selected state
    state_data = census_all[census_all['state'] == state]
    
    # Get data for 2015 and 2017
    data_2015 = state_data[state_data['year'] == 2015]
    data_2017 = state_data[state_data['year'] == 2017]
    
    # Merge to calculate changes
    pop_change = pd.merge(
        data_2015[['name', 'b01003_001e', 'b25077_001e']],
        data_2017[['name', 'b01003_001e', 'b25077_001e']],
        on='name', suffixes=('_2015', '_2017')
    )
    
    pop_change['state'] = state  # Reattach state info if needed later

    # Calculate percentage changes
    pop_change['pop_change_pct'] = (
        (pop_change['b01003_001e_2017'] - pop_change['b01003_001e_2015']) / 
        pop_change['b01003_001e_2015'] * 100
    )
    pop_change['home_value_change_pct'] = (
        (pop_change['b25077_001e_2017'] - pop_change['b25077_001e_2015']) / 
        pop_change['b25077_001e_2015'] * 100
    )

    if pop_change.empty:
        st.info("No data available for population change analysis.")
        return

    # --- SCATTER PLOT ---
    scatter_fig = px.scatter(
        pop_change, 
        x='pop_change_pct', 
        y='home_value_change_pct',
        size='b01003_001e_2017',
        hover_name='name',
        labels={
            'pop_change_pct': 'Population Change (%)', 
            'home_value_change_pct': 'Median Home Value Change (%)'
        },
        title=f'Population vs. Home Value Change in {state} (2015–2017)',
        size_max=50
    )

    # Add horizontal and vertical reference lines at 0
    scatter_fig.add_shape(
        type="line",
        x0=0, y0=pop_change['home_value_change_pct'].min(),
        x1=0, y1=pop_change['home_value_change_pct'].max(),
        line=dict(color="gray", dash="dash")
    )
    scatter_fig.add_shape(
        type="line",
        x0=pop_change['pop_change_pct'].min(), y0=0,
        x1=pop_change['pop_change_pct'].max(), y1=0,
        line=dict(color="gray", dash="dash")
    )

    # Add text annotations for notable counties
    for idx, row in pop_change.iterrows():
        if abs(row['pop_change_pct']) > 2 or abs(row['home_value_change_pct']) > 5:
            scatter_fig.add_annotation(
                x=row['pop_change_pct'],
                y=row['home_value_change_pct'],
                text=row['name'].split(',')[0],
                showarrow=True,
                arrowhead=1,
                font=dict(size=8),
                ax=20,
                ay=-20
            )

    st.plotly_chart(scatter_fig, use_container_width=True)

    # --- BAR CHARTS: TOP GROWING and DECLINING COUNTIES ---

    # Prepare top growing and declining
    top_growing = pop_change.sort_values('pop_change_pct', ascending=False).head(10)
    top_declining = pop_change.sort_values('pop_change_pct', ascending=True).head(10)

    # Create two bar charts side by side
    bar_fig = make_subplots(
        rows=1, cols=2, subplot_titles=(f"Top 10 Growing Counties in {state}", f"Top 10 Declining Counties in {state}")
    )

    # Top growing counties
    bar_fig.add_trace(
        go.Bar(
            x=[name.split(',')[0] for name in top_growing['name']],
            y=top_growing['pop_change_pct'],
            marker_color='green',
            name='Growing'
        ),
        row=1, col=1
    )

    # Top declining counties
    bar_fig.add_trace(
        go.Bar(
            x=[name.split(',')[0] for name in top_declining['name']],
            y=top_declining['pop_change_pct'],
            marker_color='red',
            name='Declining'
        ),
        row=1, col=2
    )

    bar_fig.update_layout(
        height=600, width=1000,
        showlegend=False,
        title_text=f"Top Growing and Declining Counties in {state} (2015–2017)"
    )

    bar_fig.update_xaxes(tickangle=45)
    bar_fig.update_yaxes(title_text="Population Change (%)", row=1, col=1)
    bar_fig.update_yaxes(title_text="Population Change (%)", row=1, col=2)

    st.plotly_chart(bar_fig, use_container_width=True)


@st.cache_data
def gentrification_analysis(census_data, state):
    st.subheader(f'Gentrification Analysis (2015-2017) - {state}')

    # Filter data by state
    data_2015 = census_data[(census_data['year'] == 2015) & (census_data['state'] == state)]
    data_2017 = census_data[(census_data['year'] == 2017) & (census_data['state'] == state)]

    # Merge to calculate changes
    changes = pd.merge(
        data_2015[['name', 'b19013_001e', 'b25077_001e', 'state']],
        data_2017[['name', 'b19013_001e', 'b25077_001e']],
        on='name', suffixes=('_2015', '_2017')
    )

    # Calculate percentage changes
    changes['income_change_pct'] = ((changes['b19013_001e_2017'] - changes['b19013_001e_2015']) /
                                    changes['b19013_001e_2015']) * 100
    changes['home_value_change_pct'] = ((changes['b25077_001e_2017'] - changes['b25077_001e_2015']) /
                                        changes['b25077_001e_2015']) * 100
    changes['gentrification_score'] = (changes['income_change_pct'] + changes['home_value_change_pct']) / 2
    changes = changes[changes['b25077_001e_2017'] > 0]
    if changes.empty:
        st.info("No data available for gentrification analysis.")
        return

    # 📊 Top gentrified counties bar chart
    top_gentrified = changes.sort_values('gentrification_score', ascending=False).head(15)
    fig_bar = px.bar(
        top_gentrified,
        x='name',
        y='gentrification_score',
        color='gentrification_score',
        color_continuous_scale='RdYlGn',
        labels={'name': 'County', 'gentrification_score': 'Gentrification Score'},
        title=f'Top 15 Counties by Gentrification Score (2015-2017) - {state}'
    )
    fig_bar.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig_bar, use_container_width=True)

    # 📈 Scatter plot: income change % vs home value change %
    fig_scatter = px.scatter(
        changes,
        x='income_change_pct',
        y='home_value_change_pct',
        size='b25077_001e_2017',
        color='gentrification_score',
        color_continuous_scale='RdYlGn',
        hover_name='name',
        labels={
            'income_change_pct': 'Income Change (%)',
            'home_value_change_pct': 'Home Value Change (%)',
            'b25077_001e_2017': 'Home Value (2017)'
        },
        title=f'Income Change vs. Home Value Change (2015-2017) - {state}',
        size_max=50,
        opacity=0.7
    )

    # Add lines for reference (x=0 and y=0)
    fig_scatter.add_shape(
        type='line',
        x0=0, y0=changes['home_value_change_pct'].min(),
        x1=0, y1=changes['home_value_change_pct'].max(),
        line=dict(color='Gray', dash='dash')
    )
    fig_scatter.add_shape(
        type='line',
        x0=changes['income_change_pct'].min(), y0=0,
        x1=changes['income_change_pct'].max(), y1=0,
        line=dict(color='Gray', dash='dash')
    )

    # Highlight high gentrification area (optional shading)
    fig_scatter.add_shape(
        type='rect',
        x0=0, y0=0, x1=20, y1=20,
        fillcolor='red',
        opacity=0.1,
        layer='below',
        line_width=0,
    )
    fig_scatter.add_annotation(
        x=10, y=10,
        text="High Gentrification",
        showarrow=False,
        font=dict(size=12)
    )

    st.plotly_chart(fig_scatter, use_container_width=True)


@st.cache_data
def choropleth_heat_map(census_county):
    with open('us-states.json', 'r') as f:
        geojson_data = json.load(f)
    # Filter data for years
    data_2015 = census_county[census_county['year'] == 2015]
    data_2017 = census_county[census_county['year'] == 2017]

    # Merge to calculate changes
    changes = pd.merge(
        data_2015[['name', 'b19013_001e', 'b25077_001e', 'state']],
        data_2017[['name', 'b19013_001e', 'b25077_001e']],
        on='name', suffixes=('_2015', '_2017')
    )

    # Calculate percentage changes
    changes['income_change_pct'] = ((changes['b19013_001e_2017'] - changes['b19013_001e_2015']) /
                                    changes['b19013_001e_2015']) * 100
    changes['home_value_change_pct'] = ((changes['b25077_001e_2017'] - changes['b25077_001e_2015']) /
                                        changes['b25077_001e_2015']) * 100
    changes['gentrification_score'] = (changes['income_change_pct'] + changes['home_value_change_pct']) / 2

    # Clean data: Remove extreme values or missing
    changes = changes.replace([np.inf, -np.inf], np.nan).dropna(subset=['gentrification_score'])

    # Plotly Choropleth Heat Map
    if geojson_data is not None:
        fig = px.choropleth_mapbox(
            changes,
            geojson=geojson_data,
            locations='state',  # now matching county names
            featureidkey='properties.name',  # geojson must have 'name' under properties
            color='gentrification_score',
            color_continuous_scale='RdYlGn',
            mapbox_style='carto-positron',
            zoom=3,
            center={"lat": 37.0902, "lon": -95.7129},
            range_color=(changes['gentrification_score'].min(), changes['gentrification_score'].max()),
            #scope="usa",
            labels={'gentrification_score': 'Gentrification Score'},
            title="Gentrification Heat Map (2015-2017)",
        )
        fig.update_geos(fitbounds="locations", visible=False)
        fig.update_layout(margin={"r":0,"t":50,"l":0,"b":0})

        st.plotly_chart(fig, use_container_width=True)

# Add similar functions for:
# - Property value choropleth
# - Population change analysis
# - Gentrification analysis 
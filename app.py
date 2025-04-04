import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import statsmodels.api as sm
from datetime import datetime, timedelta

st.set_page_config(
    page_title="Market Sentiment Analysis Dashboard",
    page_icon="📊",
    layout="wide"
)

st.markdown("""
<style>
    .main-title {
        font-size: 36px;
        font-weight: bold;
        margin-bottom: 20px;
        color: #1E88E5;
    }
    .sub-title {
        font-size: 24px;
        font-weight: bold;
        margin-top: 30px;
        margin-bottom: 10px;
        color: #333;
    }
    .metrics-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 16px;
        margin-bottom: 24px;
    }
    .metric-box {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .metric-value {
        font-size: 32px;
        font-weight: bold;
        color: #1E88E5;
        margin-bottom: 8px;
    }
    .metric-label {
        font-size: 14px;
        color: #666;
        line-height: 1.2;
    }
    .insight-card {
        background-color: #e8f4f8;
        border-left: 5px solid #1E88E5;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)

# title
st.markdown("<div class='main-title'>Market Sentiment Analysis Dashboard</div>", unsafe_allow_html=True)
st.markdown("Exploring the relationship between cryptocurrency and stock market sentiment")

# load data
@st.cache_data
def load_data():
    # load most recent crypto data file
    crypto_dir = 'crypto_fg_index'
    crypto_files = [f for f in os.listdir(crypto_dir) if f.endswith('.csv')]
    latest_crypto_file = max(crypto_files) if crypto_files else None
    
    # load most recent stock data file
    stock_dir = 'stock_fg_index'
    stock_files = [f for f in os.listdir(stock_dir) if f.endswith('.csv')]
    latest_stock_file = max(stock_files) if stock_files else None
    
    if latest_crypto_file and latest_stock_file:
        crypto_df = pd.read_csv(os.path.join(crypto_dir, latest_crypto_file))
        stock_df = pd.read_csv(os.path.join(stock_dir, latest_stock_file))
        
        crypto_df['date'] = pd.to_datetime(crypto_df['date'])
        stock_df['date'] = pd.to_datetime(stock_df['date'])
        
        return crypto_df, stock_df
    
    st.error("Data files not found!")
    return None, None

crypto_df, stock_df = load_data()

if crypto_df is not None and stock_df is not None:
    # find common date range
    min_date = max(crypto_df['date'].min(), stock_df['date'].min())
    max_date = min(crypto_df['date'].max(), stock_df['date'].max())
    
    crypto_filtered = crypto_df[(crypto_df['date'] >= min_date) & (crypto_df['date'] <= max_date)]
    stock_filtered = stock_df[(stock_df['date'] >= min_date) & (stock_df['date'] <= max_date)]
    
    # merge on date
    merged_df = pd.merge(
        crypto_filtered,
        stock_filtered,
        on='date',
        how='inner',
        suffixes=('_crypto', '_stock')
    )
    
    # date range selection on sidebar
    st.sidebar.markdown("<div class='sub-title'>Filter Options</div>", unsafe_allow_html=True)
    
    # default date range 
    default_start = max(
        merged_df['date'].min().date(),
        (merged_df['date'].max() - pd.Timedelta(days=365)).date()
    )
    
    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=[
            default_start,
            merged_df['date'].max().date()
        ],
        min_value=merged_df['date'].min().date(),
        max_value=merged_df['date'].max().date()
    )
    
    if len(date_range) == 2:
        start_date, end_date = date_range
        merged_filtered = merged_df[
            (merged_df['date'].dt.date >= start_date) & 
            (merged_df['date'].dt.date <= end_date)
        ]
    else:
        merged_filtered = merged_df
    
    # calc correlations
    correlation_period = st.sidebar.selectbox(
        "Correlation Period",
        options=["All Time", "Last 1 Year", "Last 6 Months", "Last 3 Months", "Last 1 Month"],
        index=0
    )
    
    # slice data based on period chosen
    if correlation_period == "Last 1 Year":
        corr_data = merged_df[merged_df['date'] >= (merged_df['date'].max() - pd.Timedelta(days=365))]
    elif correlation_period == "Last 6 Months":
        corr_data = merged_df[merged_df['date'] >= (merged_df['date'].max() - pd.Timedelta(days=180))]
    elif correlation_period == "Last 3 Months":
        corr_data = merged_df[merged_df['date'] >= (merged_df['date'].max() - pd.Timedelta(days=90))]
    elif correlation_period == "Last 1 Month":
        corr_data = merged_df[merged_df['date'] >= (merged_df['date'].max() - pd.Timedelta(days=30))]
    else:
        corr_data = merged_df
    
    correlation_fg = corr_data['fear_greed_index_crypto'].corr(corr_data['fear_greed_index_stock'])
    correlation_price_fg_crypto = corr_data['btc_price'].corr(corr_data['fear_greed_index_crypto'])
    correlation_price_fg_stock = corr_data['price'].corr(corr_data['fear_greed_index_stock'])
    correlation_price = corr_data['btc_price'].corr(corr_data['price'])
    
    # Use a single HTML block for metrics to ensure proper rendering
    metrics_html = f"""
    <div class="metrics-grid">
        <div class="metric-box">
            <div class="metric-value">{correlation_fg:.2f}</div>
            <div class="metric-label">Crypto Stock Sentiment Correlation</div>
        </div>
        <div class="metric-box">
            <div class="metric-value">{correlation_price:.2f}</div>
            <div class="metric-label">BTC S&P 500 Price Correlation</div>
        </div>
        <div class="metric-box">
            <div class="metric-value">{correlation_price_fg_crypto:.2f}</div>
            <div class="metric-label">BTC Price Sentiment Correlation</div>
        </div>
        <div class="metric-box">
            <div class="metric-value">{correlation_price_fg_stock:.2f}</div>
            <div class="metric-label">S&P 500 Price Sentiment Correlation</div>
        </div>
    </div>
    """
    
    st.markdown(metrics_html, unsafe_allow_html=True)
    
    st.markdown("<div class='sub-title'>Key Insights</div>", unsafe_allow_html=True)
    insights = []
    
    if abs(correlation_fg) > 0.7:
        insights.append(f"Strong {'positive' if correlation_fg > 0 else 'negative'} correlation ({correlation_fg:.2f}) between crypto and stock market sentiment, suggesting market sentiment tends to move {'together' if correlation_fg > 0 else 'in opposite directions'}.")
    elif abs(correlation_fg) > 0.3:
        insights.append(f"Moderate {'positive' if correlation_fg > 0 else 'negative'} correlation ({correlation_fg:.2f}) between crypto and stock market sentiment.")
    else:
        insights.append(f"Weak correlation ({correlation_fg:.2f}) between crypto and stock market sentiment, suggesting these markets may be driven by different factors.")
    
    if abs(correlation_price) > 0.7:
        insights.append(f"Strong {'positive' if correlation_price > 0 else 'negative'} correlation ({correlation_price:.2f}) between Bitcoin and S&P 500 prices, indicating {'potential market integration' if correlation_price > 0 else 'a potential hedging relationship'}.")
    elif abs(correlation_price) > 0.3:
        insights.append(f"Moderate {'positive' if correlation_price > 0 else 'negative'} correlation ({correlation_price:.2f}) between Bitcoin and S&P 500 prices.")
    else:
        insights.append(f"Weak correlation ({correlation_price:.2f}) between Bitcoin and S&P 500 prices, suggesting Bitcoin may still offer diversification benefits.")
    
    for insight in insights:
        st.markdown(f"<div class='insight-card'>{insight}</div>", unsafe_allow_html=True)
    
    # Fear & Greed Index Comparison
    st.markdown("<div class='sub-title'>Fear & Greed Index Comparison</div>", unsafe_allow_html=True)
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=merged_filtered['date'],
        y=merged_filtered['fear_greed_index_crypto'],
        mode='lines',
        name='Crypto Fear & Greed',
        line=dict(color='#FF9800', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=merged_filtered['date'],
        y=merged_filtered['fear_greed_index_stock'],
        mode='lines',
        name='Stock Fear & Greed',
        line=dict(color='#2196F3', width=2)
    ))
    
    fig.add_shape(
        type="line",
        x0=merged_filtered['date'].min(),
        y0=25,
        x1=merged_filtered['date'].max(),
        y1=25,
        line=dict(color="Red", width=1, dash="dash"),
    )
    
    fig.add_shape(
        type="line",
        x0=merged_filtered['date'].min(),
        y0=50,
        x1=merged_filtered['date'].max(),
        y1=50,
        line=dict(color="Gray", width=1, dash="dash"),
    )
    
    fig.add_shape(
        type="line",
        x0=merged_filtered['date'].min(),
        y0=75,
        x1=merged_filtered['date'].max(),
        y1=75,
        line=dict(color="Green", width=1, dash="dash"),
    )
    
    fig.add_annotation(
        x=merged_filtered['date'].min(),
        y=12.5,
        text="Extreme Fear",
        showarrow=False,
        font=dict(size=10, color="Red"),
        xanchor="left"
    )
    
    fig.add_annotation(
        x=merged_filtered['date'].min(),
        y=37.5,
        text="Fear",
        showarrow=False,
        font=dict(size=10, color="Red"),
        xanchor="left"
    )
    
    fig.add_annotation(
        x=merged_filtered['date'].min(),
        y=62.5,
        text="Greed",
        showarrow=False,
        font=dict(size=10, color="Green"),
        xanchor="left"
    )
    
    fig.add_annotation(
        x=merged_filtered['date'].min(),
        y=87.5,
        text="Extreme Greed",
        showarrow=False,
        font=dict(size=10, color="Green"),
        xanchor="left"
    )
    
    fig.update_layout(
        title="Crypto vs Stock Fear & Greed Indices",
        xaxis_title="Date",
        yaxis_title="Fear & Greed Index (0-100)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        height=500,
        hovermode="x unified",
        template="plotly_white"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Correlation between indices
    st.markdown("<div class='sub-title'>Sentiment Correlation Analysis</div>", unsafe_allow_html=True)
    fig_scatter = px.scatter(
        merged_filtered,
        x='fear_greed_index_stock',
        y='fear_greed_index_crypto',
        trendline='ols',
        trendline_color_override='red',
        labels={
            "fear_greed_index_stock": "Stock Market Fear & Greed Index",
            "fear_greed_index_crypto": "Crypto Fear & Greed Index"
        },
        title=f"Correlation between Stock and Crypto Sentiment (r = {correlation_fg:.2f})"
    )
    
    fig_scatter.update_traces(marker=dict(size=8, opacity=0.6))
    
    st.plotly_chart(fig_scatter, use_container_width=True)
    
    # Price and Sentiment Comparison
    st.markdown("<div class='sub-title'>Price and Sentiment Relationship</div>", unsafe_allow_html=True)
    tab1, tab2 = st.tabs(["Bitcoin & Crypto Sentiment", "S&P 500 & Stock Sentiment"])
    
    with tab1:
        # Bitcoin price and crypto sentiment
        fig_btc = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig_btc.add_trace(
            go.Scatter(
                x=merged_filtered['date'],
                y=merged_filtered['btc_price'],
                mode='lines',
                name='Bitcoin Price',
                line=dict(color='#F7931A', width=2)
            ),
            secondary_y=False
        )
        
        fig_btc.add_trace(
            go.Scatter(
                x=merged_filtered['date'],
                y=merged_filtered['fear_greed_index_crypto'],
                mode='lines',
                name='Crypto Fear & Greed',
                line=dict(color='#2196F3', width=2)
            ),
            secondary_y=True
        )
        
        fig_btc.update_layout(
            title="Bitcoin Price vs. Crypto Fear & Greed Index",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
            height=500,
            template="plotly_white"
        )
        
        fig_btc.update_xaxes(title_text="Date")
        fig_btc.update_yaxes(title_text="Bitcoin Price (USD)", secondary_y=False)
        fig_btc.update_yaxes(title_text="Fear & Greed Index (0-100)", secondary_y=True)
        
        st.plotly_chart(fig_btc, use_container_width=True)

        fig_btc_scatter = px.scatter(
            merged_filtered,
            x='fear_greed_index_crypto',
            y='btc_price',
            trendline='ols',
            labels={
                "fear_greed_index_crypto": "Crypto Fear & Greed Index",
                "btc_price": "Bitcoin Price (USD)"
            },
            title=f"Correlation between Crypto Sentiment and Bitcoin Price (r = {correlation_price_fg_crypto:.2f})"
        )
        
        fig_btc_scatter.update_traces(marker=dict(size=8, opacity=0.6, color='#F7931A'))
        
        st.plotly_chart(fig_btc_scatter, use_container_width=True)
    
    with tab2:
        # S&P 500 price and stock sentiment
        fig_sp500 = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig_sp500.add_trace(
            go.Scatter(
                x=merged_filtered['date'],
                y=merged_filtered['price'],
                mode='lines',
                name='S&P 500 Price',
                line=dict(color='#ffbf69', width=2)
            ),
            secondary_y=False
        )
        
        fig_sp500.add_trace(
            go.Scatter(
                x=merged_filtered['date'],
                y=merged_filtered['fear_greed_index_stock'],
                mode='lines',
                name='Stock Fear & Greed',
                line=dict(color='#2196F3', width=2)
            ),
            secondary_y=True
        )
        
        fig_sp500.update_layout(
            title="S&P 500 Price vs. Stock Fear & Greed Index",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
            height=500,
            template="plotly_white"
        )
        
        fig_sp500.update_xaxes(title_text="Date")
        fig_sp500.update_yaxes(title_text="S&P 500 Price (USD)", secondary_y=False)
        fig_sp500.update_yaxes(title_text="Fear & Greed Index (0-100)", secondary_y=True)
        
        st.plotly_chart(fig_sp500, use_container_width=True)
        
        fig_sp500_scatter = px.scatter(
            merged_filtered,
            x='fear_greed_index_stock',
            y='price',
            trendline='ols',
            labels={
                "fear_greed_index_stock": "Stock Fear & Greed Index",
                "price": "S&P 500 Price"
            },
            title=f"Correlation between Stock Sentiment and S&P 500 Price (r = {correlation_price_fg_stock:.2f})"
        )
        
        fig_sp500_scatter.update_traces(marker=dict(size=8, opacity=0.6, color='#1E88E5'))
        
        st.plotly_chart(fig_sp500_scatter, use_container_width=True)
    
    # Rolling correlation analysis
    # st.markdown("<div class='sub-title'>Rolling Correlation Analysis</div>", unsafe_allow_html=True)
    
    # # Window selection
    # window_size = st.select_slider(
    #     "Select Rolling Window Size (days)",
    #     options=[7, 14, 30, 60, 90, 180],
    #     value=30
    # )
    
    # # Create a dataframe sorted by date
    # rolling_df = merged_df.sort_values('date').copy()
    
    # # Calculate rolling correlations
    # rolling_df['rolling_fg_corr'] = rolling_df['fear_greed_index_crypto'].rolling(window=window_size).corr(rolling_df['fear_greed_index_stock'])
    # rolling_df['rolling_price_corr'] = rolling_df['btc_price'].rolling(window=window_size).corr(rolling_df['price'])
    
    # # Filter based on date range
    # rolling_filtered = rolling_df[
    #     (rolling_df['date'].dt.date >= date_range[0]) & 
    #     (rolling_df['date'].dt.date <= date_range[1])
    # ] if len(date_range) == 2 else rolling_df
    
    # # Plot rolling correlations
    # fig_rolling = go.Figure()
    
    # fig_rolling.add_trace(go.Scatter(
    #     x=rolling_filtered['date'],
    #     y=rolling_filtered['rolling_fg_corr'],
    #     mode='lines',
    #     name='Sentiment Correlation',
    #     line=dict(color='#FF9800', width=2)
    # ))
    
    # fig_rolling.add_trace(go.Scatter(
    #     x=rolling_filtered['date'],
    #     y=rolling_filtered['rolling_price_corr'],
    #     mode='lines',
    #     name='Price Correlation',
    #     line=dict(color='#9C27B0', width=2)
    # ))
    
    # fig_rolling.add_shape(
    #     type="line",
    #     x0=rolling_filtered['date'].min(),
    #     y0=0,
    #     x1=rolling_filtered['date'].max(),
    #     y1=0,
    #     line=dict(color="Gray", width=1, dash="dash"),
    # )
    
    # fig_rolling.update_layout(
    #     title=f"{window_size}-Day Rolling Correlation Between Crypto and Stock Markets",
    #     xaxis_title="Date",
    #     yaxis_title="Correlation Coefficient",
    #     legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    #     height=500,
    #     hovermode="x unified",
    #     template="plotly_white",
    #     yaxis=dict(range=[-1, 1])
    # )
    
    # st.plotly_chart(fig_rolling, use_container_width=True)
    
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666;'>Created by Kaydon Lim | Market Sentiment Analysis Project</div>",
        unsafe_allow_html=True
    )
else:
    st.error("Failed to load data. Please check if the data files exist.")
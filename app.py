import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Insider Trading Detection",
    page_icon="🔍",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .risk-high { 
        background-color: #ff4b4b; 
        color: white; 
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: bold;
    }
    .risk-medium { 
        background-color: #ffa500; 
        color: white;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: bold;
    }
    .risk-low { 
        background-color: #00cc96; 
        color: white;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: bold;
    }
    .metric-card {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

def generate_sample_company_data():
    """Generate sample company data if no file is uploaded"""
    companies = [
        {'Company': 'Apple Inc', 'Symbol': 'AAPL', 'Price': 150.25, 'Weight': 2.5},
        {'Company': 'Microsoft', 'Symbol': 'MSFT', 'Price': 330.50, 'Weight': 2.3},
        {'Company': 'Google', 'Symbol': 'GOOGL', 'Price': 2750.75, 'Weight': 1.8},
        {'Company': 'Amazon', 'Symbol': 'AMZN', 'Price': 3400.00, 'Weight': 1.9},
        {'Company': 'Tesla', 'Symbol': 'TSLA', 'Price': 250.80, 'Weight': 1.5},
        {'Company': 'Meta', 'Symbol': 'META', 'Price': 325.40, 'Weight': 1.2},
        {'Company': 'Netflix', 'Symbol': 'NFLX', 'Price': 410.30, 'Weight': 0.8},
        {'Company': 'NVIDIA', 'Symbol': 'NVDA', 'Price': 450.60, 'Weight': 1.7}
    ]
    return pd.DataFrame(companies)

def clean_company_data(company_data):
    """Clean and convert data types in company data"""
    df = company_data.copy()
    
    # Convert Price to float (handle commas and strings)
    if 'Price' in df.columns:
        df['Price'] = pd.to_numeric(df['Price'].astype(str).str.replace(',', ''), errors='coerce')
    
    # Convert Weight to float
    if 'Weight' in df.columns:
        df['Weight'] = pd.to_numeric(df['Weight'], errors='coerce')
    
    # Fill any missing values
    df['Price'] = df['Price'].fillna(100.0)
    df['Weight'] = df['Weight'].fillna(1.0)
    
    return df

def generate_trading_data(company_data, days=90):
    """Generate realistic trading data"""
    np.random.seed(42)
    data = []
    start_date = datetime.now() - timedelta(days=days)
    
    # Clean the company data first
    company_data_clean = clean_company_data(company_data)
    
    for day in range(days):
        current_date = start_date + timedelta(days=day)
        
        for _, company in company_data_clean.iterrows():
            # Generate 1-5 trades per company per day
            trades_today = np.random.randint(1, 6)
            
            for trade_num in range(trades_today):
                # Trader type
                trader_type = np.random.choice(['Retail', 'Institutional', 'Employee'], 
                                             p=[0.7, 0.2, 0.1])
                
                if trader_type == 'Employee':
                    position = np.random.choice(['CEO', 'CFO', 'Director', 'VP'])
                    volume = np.random.randint(100, 5000)
                else:
                    position = None
                    volume = np.random.randint(1000, 50000)
                
                # Ensure base_price is float
                base_price = float(company['Price'])
                
                # Price movement based on company weight
                price_change = np.random.normal(0, base_price * 0.02)
                trade_price = max(1.0, base_price + price_change)
                
                # Suspicious activity detection
                is_suspicious = False
                suspicious_reason = None
                
                # 5% chance of suspicious activity
                if np.random.random() < 0.05:
                    is_suspicious = True
                    suspicious_reason = np.random.choice([
                        "Trade before earnings",
                        "Unusual volume pattern", 
                        "Multiple employee trades",
                        "Price surge before news"
                    ])
                
                # Calculate risk score
                risk_factors = 0
                if trader_type == 'Employee':
                    risk_factors += 2
                if volume > 10000:
                    risk_factors += 1
                if is_suspicious:
                    risk_factors += 3
                if float(company['Weight']) > 2.0:
                    risk_factors += 1
                
                risk_score = min(10, risk_factors * 2)
                
                data.append({
                    'trade_id': f"TR{company['Symbol']}{day:03d}{trade_num}",
                    'date': current_date,
                    'trader_id': f"T{np.random.randint(1000,9999)}",
                    'trader_name': f"Trader_{np.random.randint(1000,9999)}",
                    'trader_type': trader_type,
                    'position': position,
                    'company': company['Company'],
                    'symbol': company['Symbol'],
                    'trade_type': np.random.choice(['BUY', 'SELL']),
                    'volume': volume,
                    'price': round(trade_price, 2),
                    'trade_value': round(volume * trade_price, 2),
                    'is_suspicious': is_suspicious,
                    'suspicious_reason': suspicious_reason,
                    'risk_score': risk_score
                })
    
    return pd.DataFrame(data)

def main():
    st.markdown('<h1 class="main-header">🔍 Insider Trading Detection Platform</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    <div style='text-align: center; margin-bottom: 2rem;'>
        <h3>Detect Suspicious Trading Patterns Using Data Analytics</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("📊 Data Configuration")
    
    # File upload
    uploaded_file = st.sidebar.file_uploader("Upload Company Data (CSV)", type=['csv'])
    
    if uploaded_file is not None:
        try:
            company_data = pd.read_csv(uploaded_file)
            # Clean the uploaded data
            company_data = clean_company_data(company_data)
            st.sidebar.success(f"✅ Loaded {len(company_data)} companies")
        except Exception as e:
            st.sidebar.error(f"Error loading file: {e}")
            company_data = generate_sample_company_data()
    else:
        company_data = generate_sample_company_data()
        st.sidebar.info("📝 Using sample data. Upload a CSV for your own analysis.")
    
    # Show company data preview
    st.sidebar.subheader("Company Data Preview")
    st.sidebar.dataframe(company_data.head(3))
    
    # Analysis parameters
    st.sidebar.title("⚙️ Analysis Settings")
    analysis_days = st.sidebar.slider("Analysis Period (Days)", 30, 180, 90)
    risk_threshold = st.sidebar.slider("Risk Alert Threshold", 1, 10, 5)
    
    if st.sidebar.button("🚀 Generate Analysis"):
        with st.spinner("Generating trading data and analyzing patterns..."):
            trading_df = generate_trading_data(company_data, analysis_days)
            st.session_state.trading_df = trading_df
            st.session_state.company_data = company_data
    
    # Check if we have data to display
    if 'trading_df' not in st.session_state:
        st.info("👈 **Configure your settings in the sidebar and click 'Generate Analysis' to start!**")
        
        # Show sample company data
        st.subheader("📋 Company Data Overview")
        st.dataframe(company_data)
        
        # Data validation info
        st.subheader("🔧 Data Validation")
        st.write("**Price Statistics:**")
        st.write(f"- Min: ${company_data['Price'].min():.2f}")
        st.write(f"- Max: ${company_data['Price'].max():.2f}")
        st.write(f"- Mean: ${company_data['Price'].mean():.2f}")
        
        st.write("**Weight Statistics:**")
        st.write(f"- Min: {company_data['Weight'].min():.2f}")
        st.write(f"- Max: {company_data['Weight'].max():.2f}")
        st.write(f"- Mean: {company_data['Weight'].mean():.2f}")
        return
    
    # Access the data
    trading_df = st.session_state.trading_df
    company_data = st.session_state.company_data
    
    # Dashboard Metrics
    st.markdown("## 📈 Executive Dashboard")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_trades = len(trading_df)
        st.markdown(f'<div class="metric-card"><h3>Total Trades</h3><h2>{total_trades:,}</h2></div>', 
                   unsafe_allow_html=True)
    
    with col2:
        suspicious_trades = trading_df['is_suspicious'].sum()
        st.markdown(f'<div class="metric-card"><h3>Suspicious Flags</h3><h2>{suspicious_trades}</h2></div>', 
                   unsafe_allow_html=True)
    
    with col3:
        high_risk = len(trading_df[trading_df['risk_score'] >= risk_threshold])
        st.markdown(f'<div class="metric-card"><h3>High Risk Alerts</h3><h2>{high_risk}</h2></div>', 
                   unsafe_allow_html=True)
    
    with col4:
        companies = trading_df['company'].nunique()
        st.markdown(f'<div class="metric-card"><h3>Companies</h3><h2>{companies}</h2></div>', 
                   unsafe_allow_html=True)
    
    # Tabs for different analyses
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Risk Alerts", "📊 Trends", "🏢 Company View", "📈 Data Explorer"])
    
    with tab1:
        st.markdown("## 🚨 High-Risk Trading Alerts")
        
        high_risk_df = trading_df[trading_df['risk_score'] >= risk_threshold].sort_values('risk_score', ascending=False)
        
        if not high_risk_df.empty:
            st.write(f"**Found {len(high_risk_df)} high-risk trades (risk ≥ {risk_threshold}):**")
            
            for idx, (_, trade) in enumerate(high_risk_df.head(20).iterrows()):
                risk_class = "risk-high" if trade['risk_score'] >= 8 else "risk-medium"
                
                with st.container():
                    col1, col2, col3 = st.columns([3, 2, 1])
                    
                    with col1:
                        st.write(f"**{trade['trader_name']}** ({trade['position'] or trade['trader_type']})")
                        st.write(f"*{trade['company']} ({trade['symbol']}) - {trade['date'].strftime('%Y-%m-%d')}*")
                        if trade['suspicious_reason']:
                            st.write(f"🔍 *{trade['suspicious_reason']}*")
                    
                    with col2:
                        st.write(f"**{trade['trade_type']}** {trade['volume']:,} shares")
                        st.write(f"Value: ${trade['trade_value']:,.0f}")
                    
                    with col3:
                        st.markdown(f'<span class="{risk_class}">Risk: {trade["risk_score"]}/10</span>', 
                                   unsafe_allow_html=True)
                    
                    st.divider()
        else:
            st.success("✅ No high-risk trades detected!")
    
    with tab2:
        st.markdown("## 📊 Trading Pattern Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Risk distribution
            fig = px.histogram(trading_df, x='risk_score', 
                              title='Distribution of Risk Scores',
                              nbins=20)
            st.plotly_chart(fig, use_container_width=True)
            
            # Trader type analysis
            trader_summary = trading_df.groupby('trader_type').agg({
                'trade_id': 'count',
                'risk_score': 'mean'
            }).reset_index()
            
            fig = px.pie(trader_summary, values='trade_id', names='trader_type',
                        title='Trades by Trader Type')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Volume trends over time
            daily_volume = trading_df.groupby(trading_df['date'].dt.date).agg({
                'volume': 'sum',
                'risk_score': 'mean'
            }).reset_index()
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=daily_volume['date'], y=daily_volume['volume'],
                                   mode='lines', name='Daily Volume',
                                   line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=daily_volume['date'], y=daily_volume['risk_score'] * 10000,
                                   mode='markers', name='Risk Score (scaled)',
                                   marker=dict(color='red', size=6)))
            
            fig.update_layout(title='Trading Volume & Risk Over Time',
                             xaxis_title='Date')
            st.plotly_chart(fig, use_container_width=True)
            
            # Company risk comparison
            company_risk = trading_df.groupby('company').agg({
                'risk_score': 'mean',
                'trade_value': 'sum'
            }).reset_index()
            
            fig = px.bar(company_risk.nlargest(10, 'risk_score'), 
                        x='company', y='risk_score',
                        title='Top 10 Companies by Average Risk Score')
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.markdown("## 🏢 Company-Specific Analysis")
        
        selected_company = st.selectbox("Select Company", trading_df['company'].unique())
        
        company_trades = trading_df[trading_df['company'] == selected_company]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Total Trades", len(company_trades))
            st.metric("Average Risk Score", f"{company_trades['risk_score'].mean():.2f}")
            st.metric("Suspicious Trades", company_trades['is_suspicious'].sum())
            
            # Top risky trades for this company
            st.subheader("Top Risky Trades")
            risky_company = company_trades.nlargest(5, 'risk_score')[['date', 'trader_type', 'volume', 'risk_score']]
            st.dataframe(risky_company)
        
        with col2:
            # Company trading pattern
            fig = px.scatter(company_trades, x='date', y='volume',
                            color='risk_score', size='volume',
                            title=f'Trading Pattern for {selected_company}',
                            color_continuous_scale='viridis')
            st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.markdown("## 📈 Data Explorer")
        
        # Filters
        col1, col2 = st.columns(2)
        
        with col1:
            min_risk = st.slider("Minimum Risk Score", 0, 10, 0)
            selected_trader = st.selectbox("Trader Type", 
                                         ['All'] + list(trading_df['trader_type'].unique()))
        
        with col2:
            selected_company = st.selectbox("Company", 
                                          ['All'] + list(trading_df['company'].unique()))
        
        # Apply filters
        filtered_data = trading_df[trading_df['risk_score'] >= min_risk]
        
        if selected_trader != 'All':
            filtered_data = filtered_data[filtered_data['trader_type'] == selected_trader]
        
        if selected_company != 'All':
            filtered_data = filtered_data[filtered_data['company'] == selected_company]
        
        st.write(f"**Filtered Results: {len(filtered_data)} trades**")
        st.dataframe(filtered_data.sort_values('risk_score', ascending=False))
        
        # Export data
        csv = filtered_data.to_csv(index=False)
        st.download_button(
            label="📥 Download Filtered Data",
            data=csv,
            file_name="insider_trading_analysis.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()
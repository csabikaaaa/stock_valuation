"""
Enhanced Stock Valuation Application - Version 2.0
Now includes Peter Lynch, Warren Buffett, Benjamin Graham, and Hedge Fund methods
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import numpy as np

# Enhanced modules
from data_fetcher import StockDataFetcher
from valuation_methods import EnhancedValuationCalculator
from market_data import MarketAveragesFetcher

# Page configuration
st.set_page_config(
    page_title="Enhanced Stock Valuation Tool",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS for legendary investor themes
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(45deg, #1E3A8A, #059669, #DC2626);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    .legendary-section {
        background: linear-gradient(135deg, #F8FAFC 0%, #E2E8F0 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        border-left: 5px solid #3B82F6;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .lynch-method { border-left-color: #059669; }
    .buffett-method { border-left-color: #DC2626; }
    .graham-method { border-left-color: #7C3AED; }
    .hedge-fund-method { border-left-color: #F59E0B; }
    .method-title {
        font-size: 1.2rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .status-excellent { color: #059669; font-weight: bold; }
    .status-good { color: #0891B2; font-weight: bold; }
    .status-fair { color: #D97706; font-weight: bold; }
    .status-poor { color: #DC2626; font-weight: bold; }
    .investor-quote {
        font-style: italic;
        color: #64748B;
        border-left: 3px solid #CBD5E1;
        padding-left: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def main():
    st.markdown('<h1 class="main-header">🚀 Enhanced Stock Valuation Tool</h1>', unsafe_allow_html=True)

    # Legendary investor quotes
    quotes = [
        '"Know what you own, and know why you own it." - Peter Lynch"',
        '"Price is what you pay. Value is what you get." - Warren Buffett"', 
        '"The intelligent investor is a realist who sells to optimists and buys from pessimists." - Benjamin Graham"'
    ]

    selected_quote = np.random.choice(quotes)
    st.markdown(f'<p class="investor-quote">{selected_quote}</p>', unsafe_allow_html=True)

    # Initialize session state
    if 'stock_data' not in st.session_state:
        st.session_state.stock_data = None
    if 'valuation_results' not in st.session_state:
        st.session_state.valuation_results = None

    # Enhanced sidebar
    with st.sidebar:
        st.header("🎯 Enhanced Stock Analysis")

        # Stock symbol input
        symbol = st.text_input(
            "Enter Stock Symbol", 
            value="AAPL", 
            help="Examples: AAPL, MSFT, GOOGL, TSLA, BRK-B"
        ).upper()

        # Analysis type selector
        analysis_type = st.selectbox(
            "Analysis Focus",
            ["Complete Analysis", "Value Investing", "Growth Investing", "Hedge Fund Style"],
            help="Choose analysis approach"
        )

        # Period selection
        period_options = {
            "1 Month": "1mo",
            "3 Months": "3mo", 
            "6 Months": "6mo",
            "1 Year": "1y",
            "2 Years": "2y",
            "5 Years": "5y"
        }
        selected_period = st.selectbox("Historical Data Period", list(period_options.keys()), index=3)

        # Enhanced parameters
        st.subheader("🧮 Valuation Parameters")

        col1, col2 = st.columns(2)
        with col1:
            growth_rate = st.slider("Growth Rate (%)", 0.0, 25.0, 8.0, 0.5)
            discount_rate = st.slider("Discount Rate (%)", 5.0, 20.0, 10.0, 0.5)

        with col2:
            margin_of_safety = st.slider("Margin of Safety (%)", 10, 50, 25, 5)
            risk_tolerance = st.selectbox("Risk Level", ["Conservative", "Moderate", "Aggressive"])

        st.session_state.growth_rate = growth_rate
        st.session_state.discount_rate = discount_rate
        st.session_state.margin_of_safety = margin_of_safety
        st.session_state.risk_tolerance = risk_tolerance

        # Fetch data button
        if st.button("🔄 Analyze Stock", type="primary"):
            with st.spinner("Fetching enhanced analysis..."):
                fetcher = StockDataFetcher()
                stock_data = fetcher.get_stock_data(symbol, period_options[selected_period])

                if stock_data:
                    st.session_state.stock_data = stock_data
                    st.session_state.symbol = symbol
                    st.session_state.analysis_type = analysis_type

                    # Calculate enhanced valuations
                    calculator = EnhancedValuationCalculator(stock_data)
                    st.session_state.valuation_results = calculator.calculate_all_valuations()
                    st.success("Enhanced analysis completed!")
                else:
                    st.error("Failed to fetch data. Please check the symbol.")

    # Main content area
    if st.session_state.stock_data:
        display_enhanced_analysis()
    else:
        display_enhanced_welcome()

def display_enhanced_welcome():
    st.markdown("""
    ## Welcome to the Enhanced Stock Valuation Tool! 🚀

    ### 🌟 Now Featuring Legendary Investor Methods:

    #### 📈 Peter Lynch Approaches:
    - **GARP Strategy**: Growth At Reasonable Price analysis
    - **Lynch Fair Value**: (Growth Rate + Dividend Yield) ÷ P/E Ratio
    - **PEG Ratio**: Lynch's favorite metric for growth stocks

    #### 💰 Warren Buffett Methods:
    - **Buffett Criteria**: 10-point investment checklist
    - **Intrinsic Value**: Conservative DCF with owner earnings
    - **Margin of Safety**: Buy with significant discount to intrinsic value

    #### 📊 Benjamin Graham Techniques:
    - **Net-Net Working Capital**: Graham's ultra-conservative approach
    - **Defensive Investor**: 7 criteria for safe investments
    - **Graham Formula**: Statistical bargain hunting

    #### 🎯 Joel Greenblatt's Magic Formula:
    - **Earnings Yield**: EBIT ÷ Enterprise Value
    - **Return on Capital**: Quality business identification
    - **Combined Ranking**: "Good companies at cheap prices"

    #### 🏦 Hedge Fund Strategies:
    - **Momentum Factors**: Multi-timeframe momentum analysis
    - **Quality Scores**: Comprehensive business quality metrics
    - **Mean Reversion**: Contrarian opportunities identification
    - **Multi-Factor Models**: Quantitative factor combinations

    ### 🎯 Analysis Types Available:
    - **Complete Analysis**: All methods (20+ valuation approaches)
    - **Value Investing**: Focus on Graham, Buffett, and value metrics
    - **Growth Investing**: Emphasize Lynch, growth metrics, and momentum
    - **Hedge Fund Style**: Quantitative factors and systematic approaches

    ### 💡 Enhanced Features:
    - **Risk-Adjusted Returns**: Sharpe ratios and volatility analysis
    - **Sector Comparisons**: Industry-relative valuations
    - **Time-Series Analysis**: Historical performance patterns
    - **Scenario Analysis**: Multiple growth and discount rate assumptions

    **Ready to analyze like the legends? Enter a stock symbol and choose your approach!**
    """)

def display_enhanced_analysis():
    stock_data = st.session_state.stock_data
    symbol = st.session_state.symbol
    analysis_type = getattr(st.session_state, 'analysis_type', 'Complete Analysis')

    # Enhanced company header
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("Company", stock_data['info'].get('longName', symbol))
    with col2:
        current_price = stock_data['info'].get('currentPrice', 0)
        prev_close = stock_data['info'].get('previousClose', current_price)
        price_change = current_price - prev_close
        st.metric("Price", f"${current_price:.2f}", f"{price_change:.2f}")
    with col3:
        market_cap = stock_data['info'].get('marketCap', 0)
        market_cap_display = format_large_number(market_cap)
        st.metric("Market Cap", market_cap_display)
    with col4:
        sector = stock_data['info'].get('sector', 'N/A')
        st.metric("Sector", sector)
    with col5:
        pe_ratio = stock_data['info'].get('trailingPE', 0)
        st.metric("P/E Ratio", f"{pe_ratio:.1f}" if pe_ratio else "N/A")

    # Enhanced tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview", 
        "🏆 Legendary Methods", 
        "📈 Charts & Technical", 
        "🧮 All Valuations",
        "💾 Export & Summary"
    ])

    with tab1:
        display_enhanced_overview(stock_data)

    with tab2:
        display_legendary_methods()

    with tab3:
        display_enhanced_charts(stock_data)

    with tab4:
        display_all_valuations()

    with tab5:
        display_enhanced_export()

def display_enhanced_overview(stock_data):
    st.subheader("📈 Enhanced Company Overview")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Company Fundamentals")
        info = stock_data['info']

        # Key metrics with color coding
        metrics = {
            'Beta': info.get('beta', 0),
            'ROE': f"{info.get('returnOnEquity', 0)*100:.1f}%" if info.get('returnOnEquity') else 'N/A',
            'ROA': f"{info.get('returnOnAssets', 0)*100:.1f}%" if info.get('returnOnAssets') else 'N/A',
            'Profit Margin': f"{info.get('profitMargins', 0)*100:.1f}%" if info.get('profitMargins') else 'N/A',
            'Debt/Equity': f"{info.get('debtToEquity', 0):.1f}%" if info.get('debtToEquity') else 'N/A',
            'Current Ratio': f"{info.get('currentRatio', 0):.2f}" if info.get('currentRatio') else 'N/A',
        }

        for key, value in metrics.items():
            st.write(f"**{key}**: {value}")

    with col2:
        st.markdown("### Quick Assessment")

        # Quick assessment based on key metrics
        assessment = []

        # Financial strength
        debt_ratio = info.get('debtToEquity', 100)
        if debt_ratio < 30:
            assessment.append("🟢 Strong Balance Sheet")
        elif debt_ratio < 60:
            assessment.append("🟡 Moderate Debt Level")
        else:
            assessment.append("🔴 High Debt Concern")

        # Profitability
        roe = info.get('returnOnEquity', 0)
        if roe > 0.15:
            assessment.append("🟢 Highly Profitable")
        elif roe > 0.10:
            assessment.append("🟡 Profitable")
        else:
            assessment.append("🔴 Low Profitability")

        # Valuation
        pe_ratio = info.get('trailingPE', 100)
        if pe_ratio < 15:
            assessment.append("🟢 Attractive Valuation")
        elif pe_ratio < 25:
            assessment.append("🟡 Fair Valuation")
        else:
            assessment.append("🔴 Expensive Valuation")

        # Growth
        revenue_growth = info.get('revenueGrowth', 0)
        if revenue_growth > 0.10:
            assessment.append("🟢 Strong Growth")
        elif revenue_growth > 0.05:
            assessment.append("🟡 Moderate Growth")
        else:
            assessment.append("🔴 Slow Growth")

        for item in assessment:
            st.write(item)

    info = stock_data['info']
    industry_name = info.get('industry', '')
    sector_name = info.get('sector', '')
    industry_metrics = MarketAveragesFetcher.get_industry_metrics(industry_name)
    sector_metrics = MarketAveragesFetcher.get_sector_metrics(sector_name)

    st.markdown("### Industry & Sector Benchmarks")
    benchmark_col1, benchmark_col2 = st.columns(2)

    with benchmark_col1:
        st.markdown(
            f"**Industry:** {industry_metrics.get('name', industry_name or 'N/A')}"
        )
        ind_metrics_cols = st.columns(2)
        with ind_metrics_cols[0]:
            st.metric("P/E", format_ratio(industry_metrics.get('pe')))
        with ind_metrics_cols[1]:
            st.metric("P/S", format_ratio(industry_metrics.get('ps')))
        if industry_metrics.get('source'):
            st.caption(f"Source: {industry_metrics['source']}")

    with benchmark_col2:
        st.markdown(
            f"**Sector:** {sector_metrics.get('name', sector_name or 'N/A')}"
        )
        sec_metrics_cols = st.columns(2)
        with sec_metrics_cols[0]:
            st.metric("P/E", format_ratio(sector_metrics.get('pe')))
        with sec_metrics_cols[1]:
            st.metric("P/S", format_ratio(sector_metrics.get('ps')))
        if sector_metrics.get('source'):
            st.caption(f"Source: {sector_metrics['source']}")

def display_legendary_methods():
    if not st.session_state.valuation_results:
        st.info("No valuation data available. Please analyze a stock first.")
        return

    results = st.session_state.valuation_results
    analysis_type = getattr(st.session_state, 'analysis_type', 'Complete Analysis')

    # Peter Lynch Methods
    st.markdown('<div class="legendary-section lynch-method">', unsafe_allow_html=True)
    st.markdown('<div class="method-title">📈 Peter Lynch Methods</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        garp_result = results.get('Peter Lynch GARP', {})
        if garp_result.get('value') != 'N/A':
            status_class = get_status_class(garp_result.get('status', ''))
            st.markdown(f"**GARP Analysis**: <span class='{status_class}'>{garp_result.get('value', 'N/A')} ({garp_result.get('status', 'N/A')})</span>", unsafe_allow_html=True)
            st.write(garp_result.get('description', ''))
        else:
            st.write("**GARP Analysis**: Data not available")

    with col2:
        lynch_fair = results.get('Lynch Fair Value', {})
        if lynch_fair.get('value') != 'N/A':
            status_class = get_status_class(lynch_fair.get('status', ''))
            st.markdown(f"**Fair Value**: <span class='{status_class}'>{lynch_fair.get('value', 'N/A')} ({lynch_fair.get('status', 'N/A')})</span>", unsafe_allow_html=True)
            st.write(lynch_fair.get('description', ''))
        else:
            st.write("**Fair Value**: Data not available")

    st.markdown('</div>', unsafe_allow_html=True)

    # Warren Buffett Methods
    st.markdown('<div class="legendary-section buffett-method">', unsafe_allow_html=True)
    st.markdown('<div class="method-title">💰 Warren Buffett Methods</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        buffett_score = results.get('Warren Buffett Score', {})
        if buffett_score.get('value') != 'N/A':
            status_class = get_status_class(buffett_score.get('status', ''))
            st.markdown(f"**Buffett Criteria**: <span class='{status_class}'>{buffett_score.get('value', 'N/A')} ({buffett_score.get('status', 'N/A')})</span>", unsafe_allow_html=True)

            # Show criteria details
            criteria_met = buffett_score.get('criteria_met', [])
            if criteria_met:
                st.write("✅ Criteria Met:")
                for criterion in criteria_met[:3]:  # Show first 3
                    st.write(f"  • {criterion}")
        else:
            st.write("**Buffett Criteria**: Data not available")

    with col2:
        buffett_iv = results.get('Buffett Intrinsic Value', {})
        if buffett_iv.get('value') != 'N/A':
            status_class = get_status_class(buffett_iv.get('status', ''))
            st.markdown(f"**Intrinsic Value**: <span class='{status_class}'>{buffett_iv.get('value', 'N/A')} ({buffett_iv.get('status', 'N/A')})</span>", unsafe_allow_html=True)

            margin = buffett_iv.get('margin_of_safety', 0)
            margin_color = "🟢" if margin > 15 else "🟡" if margin > 0 else "🔴"
            st.write(f"{margin_color} Margin of Safety: {margin:.1f}%")
        else:
            st.write("**Intrinsic Value**: Data not available")

    st.markdown('</div>', unsafe_allow_html=True)

    # Benjamin Graham Methods
    st.markdown('<div class="legendary-section graham-method">', unsafe_allow_html=True)
    st.markdown('<div class="method-title">📊 Benjamin Graham Methods</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        net_net = results.get('Benjamin Graham Net-Net', {})
        if net_net.get('value') != 'N/A':
            status_class = get_status_class(net_net.get('status', ''))
            st.markdown(f"**Net-Net Value**: <span class='{status_class}'>{net_net.get('value', 'N/A')} ({net_net.get('status', 'N/A')})</span>", unsafe_allow_html=True)
            st.write(net_net.get('description', ''))
        else:
            st.write("**Net-Net Value**: Data not available")

    with col2:
        defensive = results.get('Graham Defensive', {})
        if defensive.get('value') != 'N/A':
            status_class = get_status_class(defensive.get('status', ''))
            st.markdown(f"**Defensive Criteria**: <span class='{status_class}'>{defensive.get('value', 'N/A')} ({defensive.get('status', 'N/A')})</span>", unsafe_allow_html=True)
            st.write(defensive.get('description', ''))
        else:
            st.write("**Defensive Criteria**: Data not available")

    st.markdown('</div>', unsafe_allow_html=True)

    # Magic Formula & Hedge Fund Methods
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="legendary-section">', unsafe_allow_html=True)
        st.markdown('<div class="method-title">🎯 Magic Formula (Greenblatt)</div>', unsafe_allow_html=True)

        magic = results.get('Magic Formula (Greenblatt)', {})
        if magic.get('value') != 'N/A':
            status_class = get_status_class(magic.get('status', ''))
            st.markdown(f"**Magic Score**: <span class='{status_class}'>{magic.get('value', 'N/A')} ({magic.get('status', 'N/A')})</span>", unsafe_allow_html=True)
            st.write(magic.get('description', ''))
        else:
            st.write("**Magic Formula**: Data not available")

        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="legendary-section hedge-fund-method">', unsafe_allow_html=True)
        st.markdown('<div class="method-title">🏦 Hedge Fund Factors</div>', unsafe_allow_html=True)

        multi_factor = results.get('Multi-Factor Model', {})
        if multi_factor.get('value') != 'N/A':
            status_class = get_status_class(multi_factor.get('status', ''))
            st.markdown(f"**Factor Score**: <span class='{status_class}'>{multi_factor.get('value', 'N/A')} ({multi_factor.get('status', 'N/A')})</span>", unsafe_allow_html=True)
            st.write(multi_factor.get('description', ''))
        else:
            st.write("**Factor Analysis**: Data not available")

        st.markdown('</div>', unsafe_allow_html=True)

def display_enhanced_charts(stock_data):
    st.subheader("📈 Advanced Charts & Technical Analysis")

    history = stock_data['history']

    if history is not None and not history.empty:
        # Enhanced price chart with more indicators
        fig = go.Figure()

        # Candlestick chart
        fig.add_trace(go.Candlestick(
            x=history.index,
            open=history['Open'],
            high=history['High'],
            low=history['Low'],
            close=history['Close'],
            name='Price'
        ))

        # Multiple moving averages
        if len(history) >= 20:
            history['MA20'] = history['Close'].rolling(window=20).mean()
            fig.add_trace(go.Scatter(
                x=history.index,
                y=history['MA20'],
                mode='lines',
                name='MA 20',
                line=dict(color='orange', width=1)
            ))

        if len(history) >= 50:
            history['MA50'] = history['Close'].rolling(window=50).mean()
            fig.add_trace(go.Scatter(
                x=history.index,
                y=history['MA50'],
                mode='lines',
                name='MA 50',
                line=dict(color='blue', width=1)
            ))

        if len(history) >= 200:
            history['MA200'] = history['Close'].rolling(window=200).mean()
            fig.add_trace(go.Scatter(
                x=history.index,
                y=history['MA200'],
                mode='lines',
                name='MA 200',
                line=dict(color='red', width=2)
            ))

        # Bollinger Bands
        if len(history) >= 20:
            bb_upper = history['MA20'] + (history['Close'].rolling(20).std() * 2)
            bb_lower = history['MA20'] - (history['Close'].rolling(20).std() * 2)

            fig.add_trace(go.Scatter(
                x=history.index,
                y=bb_upper,
                mode='lines',
                name='BB Upper',
                line=dict(color='gray', width=1, dash='dash'),
                showlegend=False
            ))

            fig.add_trace(go.Scatter(
                x=history.index,
                y=bb_lower,
                mode='lines',
                name='BB Lower',
                line=dict(color='gray', width=1, dash='dash'),
                fill='tonexty',
                fillcolor='rgba(128,128,128,0.1)',
                showlegend=False
            ))

        fig.update_layout(
            title=f'{st.session_state.symbol} - Advanced Technical Analysis',
            xaxis_title='Date',
            yaxis_title='Price ($)',
            height=600,
            xaxis_rangeslider_visible=False
        )

        st.plotly_chart(fig, use_container_width=True)

        # Technical indicators summary
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            # RSI calculation
            delta = history['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs)).iloc[-1] if len(history) >= 14 else 50

            rsi_status = "🟢 Oversold" if rsi < 30 else "🔴 Overbought" if rsi > 70 else "🟡 Neutral"
            st.metric("RSI (14)", f"{rsi:.1f}", rsi_status)

        with col2:
            # Volatility
            daily_returns = history['Close'].pct_change().dropna()
            volatility = daily_returns.std() * np.sqrt(252) * 100
            vol_status = "🟢 Low" if volatility < 20 else "🔴 High" if volatility > 40 else "🟡 Normal"
            st.metric("Volatility (Annual)", f"{volatility:.1f}%", vol_status)

        with col3:
            # Support/Resistance
            recent_low = history['Low'].rolling(20).min().iloc[-1]
            recent_high = history['High'].rolling(20).max().iloc[-1]
            current_price = history['Close'].iloc[-1]

            position = (current_price - recent_low) / (recent_high - recent_low) * 100
            position_status = "🟢 Near Support" if position < 25 else "🔴 Near Resistance" if position > 75 else "🟡 Mid-Range"
            st.metric("Position in Range", f"{position:.0f}%", position_status)

        with col4:
            # Trend strength
            if len(history) >= 50:
                ma20 = history['MA20'].iloc[-1]
                ma50 = history['MA50'].iloc[-1]
                trend = "🟢 Bullish" if ma20 > ma50 else "🔴 Bearish"
            else:
                trend = "🟡 Neutral"
            st.metric("Trend (MA20 vs MA50)", trend)

def display_all_valuations():
    if not st.session_state.valuation_results:
        st.info("No valuation data available. Please analyze a stock first.")
        return

    st.subheader("🧮 Complete Valuation Analysis")

    results = st.session_state.valuation_results

    # Create comprehensive results table
    valuation_data = []

    # Group methods by category
    categories = {
        'Basic Ratios': ['P/E Ratio', 'P/B Ratio', 'P/S Ratio', 'PEG Ratio', 'Dividend Yield'],
        'Cash Flow': ['P/CF Ratio', 'EV/EBITDA', 'Asset-Based Value'],
        'Peter Lynch': ['Peter Lynch GARP', 'Lynch Fair Value'],
        'Warren Buffett': ['Warren Buffett Score', 'Buffett Intrinsic Value'],
        'Benjamin Graham': ['Benjamin Graham Net-Net', 'Graham Defensive'],
        'Modern Methods': ['Magic Formula (Greenblatt)'],
        'Hedge Fund Factors': ['Momentum Score', 'Quality Score', 'Mean Reversion', 'Multi-Factor Model']
    }

    for category, methods in categories.items():
        st.markdown(f"### {category}")

        category_data = []
        for method in methods:
            if method in results:
                result = results[method]
                if isinstance(result, dict):
                    category_data.append({
                        'Method': method,
                        'Value': result.get('value', 'N/A'),
                        'Status': result.get('status', 'N/A'),
                        'Description': result.get('description', '')
                    })

        if category_data:
            df_category = pd.DataFrame(category_data)

            # Style the dataframe
            def style_status(val):
                if 'Buy' in val or 'Undervalued' in val or 'Strong' in val or 'High Quality' in val:
                    return 'background-color: #D1FAE5; color: #065F46;'
                elif 'Sell' in val or 'Overvalued' in val or 'Poor' in val or 'Avoid' in val:
                    return 'background-color: #FEE2E2; color: #991B1B;'
                elif 'Fair' in val or 'Good' in val or 'Moderate' in val:
                    return 'background-color: #FEF3C7; color: #92400E;'
                else:
                    return ''

            styled_df = df_category.style.applymap(style_status, subset=['Status'])
            st.dataframe(styled_df, use_container_width=True, hide_index=True)
        else:
            st.write("No data available for this category.")

    # Overall recommendation
    st.markdown("### 🎯 Overall Investment Recommendation")

    # Calculate overall score based on all methods
    buy_signals = 0
    sell_signals = 0
    neutral_signals = 0
    total_signals = 0

    for method, result in results.items():
        if isinstance(result, dict) and result.get('status') != 'N/A':
            status = result.get('status', '').lower()
            total_signals += 1

            if any(word in status for word in ['buy', 'undervalued', 'strong', 'excellent', 'good']):
                buy_signals += 1
            elif any(word in status for word in ['sell', 'overvalued', 'poor', 'avoid', 'weak']):
                sell_signals += 1
            else:
                neutral_signals += 1

    if total_signals > 0:
        buy_pct = buy_signals / total_signals
        sell_pct = sell_signals / total_signals

        if buy_pct > 0.6:
            overall_rec = "🟢 STRONG BUY"
            rec_color = "#059669"
        elif buy_pct > 0.4:
            overall_rec = "🟢 BUY"
            rec_color = "#10B981"
        elif sell_pct > 0.6:
            overall_rec = "🔴 STRONG SELL"
            rec_color = "#DC2626"
        elif sell_pct > 0.4:
            overall_rec = "🔴 SELL"
            rec_color = "#EF4444"
        else:
            overall_rec = "🟡 HOLD"
            rec_color = "#F59E0B"

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown(f"<h3 style='color: {rec_color};'>{overall_rec}</h3>", unsafe_allow_html=True)

        with col2:
            st.metric("Buy Signals", f"{buy_signals}/{total_signals}")

        with col3:
            st.metric("Sell Signals", f"{sell_signals}/{total_signals}")

        with col4:
            st.metric("Neutral Signals", f"{neutral_signals}/{total_signals}")

        st.markdown(f"""
        **Analysis Summary:**
        - **{buy_pct:.1%}** of methods suggest buying
        - **{sell_pct:.1%}** of methods suggest selling
        - **{neutral_signals/total_signals:.1%}** are neutral

        *Remember: This is educational analysis only. Always do your own research and consider consulting a financial advisor.*
        """)

def display_enhanced_export():
    st.subheader("💾 Enhanced Export & Analysis Summary")

    if st.session_state.stock_data and st.session_state.valuation_results:

        # Generate comprehensive analysis report
        col1, col2 = st.columns(2)

        with col1:
            if st.button("📊 Download Complete Analysis", type="primary"):
                complete_analysis = generate_complete_analysis()
                st.download_button(
                    label="📥 Download Analysis Report",
                    data=complete_analysis,
                    file_name=f"{st.session_state.symbol}_complete_analysis.csv",
                    mime="text/csv"
                )

        with col2:
            if st.button("🎯 Download Legendary Methods Only"):
                legendary_analysis = generate_legendary_analysis()
                st.download_button(
                    label="📥 Download Legendary Analysis",
                    data=legendary_analysis,
                    file_name=f"{st.session_state.symbol}_legendary_methods.csv",
                    mime="text/csv"
                )

        # Show preview
        st.markdown("### Analysis Preview")
        preview_data = generate_preview_data()
        if preview_data:
            preview_df = pd.DataFrame(preview_data)
            st.dataframe(preview_df, use_container_width=True)

def generate_complete_analysis():
    """Generate complete analysis CSV"""
    data = []

    for method, result in st.session_state.valuation_results.items():
        if isinstance(result, dict):
            data.append({
                'Stock Symbol': st.session_state.symbol,
                'Method': method,
                'Value': result.get('value', 'N/A'),
                'Status': result.get('status', 'N/A'),
                'Description': result.get('description', ''),
                'Category': get_method_category(method),
                'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })

    df = pd.DataFrame(data)
    return df.to_csv(index=False)

def generate_legendary_analysis():
    """Generate legendary methods only CSV"""
    legendary_methods = [
        'Peter Lynch GARP', 'Lynch Fair Value',
        'Warren Buffett Score', 'Buffett Intrinsic Value',
        'Benjamin Graham Net-Net', 'Graham Defensive',
        'Magic Formula (Greenblatt)'
    ]

    data = []
    for method in legendary_methods:
        if method in st.session_state.valuation_results:
            result = st.session_state.valuation_results[method]
            if isinstance(result, dict):
                data.append({
                    'Stock Symbol': st.session_state.symbol,
                    'Legendary Method': method,
                    'Value': result.get('value', 'N/A'),
                    'Status': result.get('status', 'N/A'),
                    'Description': result.get('description', ''),
                    'Investor': get_investor_name(method),
                    'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                })

    df = pd.DataFrame(data)
    return df.to_csv(index=False)

def generate_preview_data():
    """Generate preview data for display"""
    preview_data = []

    key_methods = [
        'Peter Lynch GARP', 'Warren Buffett Score', 'Benjamin Graham Net-Net',
        'Magic Formula (Greenblatt)', 'Multi-Factor Model'
    ]

    for method in key_methods:
        if method in st.session_state.valuation_results:
            result = st.session_state.valuation_results[method]
            if isinstance(result, dict):
                preview_data.append({
                    'Method': method,
                    'Value': result.get('value', 'N/A'),
                    'Status': result.get('status', 'N/A')
                })

    return preview_data

def get_method_category(method):
    """Get category for a method"""
    categories = {
        'Peter Lynch GARP': 'Growth Investing',
        'Lynch Fair Value': 'Growth Investing',
        'Warren Buffett Score': 'Value Investing',
        'Buffett Intrinsic Value': 'Value Investing',
        'Benjamin Graham Net-Net': 'Deep Value',
        'Graham Defensive': 'Conservative',
        'Magic Formula (Greenblatt)': 'Quantitative',
        'Momentum Score': 'Hedge Fund',
        'Quality Score': 'Hedge Fund',
        'Mean Reversion': 'Hedge Fund',
        'Multi-Factor Model': 'Hedge Fund'
    }
    return categories.get(method, 'Basic Analysis')

def get_investor_name(method):
    """Get investor name for a method"""
    investors = {
        'Peter Lynch GARP': 'Peter Lynch',
        'Lynch Fair Value': 'Peter Lynch',
        'Warren Buffett Score': 'Warren Buffett',
        'Buffett Intrinsic Value': 'Warren Buffett',
        'Benjamin Graham Net-Net': 'Benjamin Graham',
        'Graham Defensive': 'Benjamin Graham',
        'Magic Formula (Greenblatt)': 'Joel Greenblatt'
    }
    return investors.get(method, 'Modern Quant')

def get_status_class(status):
    """Get CSS class for status"""
    status_lower = status.lower()
    if any(word in status_lower for word in ['buy', 'undervalued', 'strong', 'excellent', 'good']):
        return 'status-excellent'
    elif any(word in status_lower for word in ['sell', 'overvalued', 'poor', 'avoid', 'weak']):
        return 'status-poor'
    elif any(word in status_lower for word in ['fair', 'moderate', 'neutral']):
        return 'status-fair'
    else:
        return 'status-good'

def format_large_number(number):
    """Format large numbers for display"""
    if pd.isna(number) or number == 0:
        return "N/A"

    if abs(number) >= 1e12:
        return f"${number/1e12:.2f}T"
    elif abs(number) >= 1e9:
        return f"${number/1e9:.2f}B"
    elif abs(number) >= 1e6:
        return f"${number/1e6:.2f}M"
    elif abs(number) >= 1e3:
        return f"${number/1e3:.2f}K"
    else:
        return f"${number:.2f}"


def format_ratio(value):
    """Format ratio metrics such as P/E and P/S."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "N/A"
    return f"{value:.2f}x"


if __name__ == "__main__":
    main()

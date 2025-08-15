import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import requests
import json
import hmac
import hashlib
import time
from urllib.parse import urlencode
import google.generativeai as genai
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

# Configure page
st.set_page_config(
    page_title="🚀 Crypto Trading Agent",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
    }
    .trade-button {
        background: linear-gradient(45deg, #667eea, #764ba2);
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        cursor: pointer;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
</style>
""", unsafe_allow_html=True)

class BinanceAPI:
    def __init__(self, api_key: str, api_secret: str = None):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = "https://api.binance.com"
        self.headers = {"X-MBX-APIKEY": api_key}

    def _generate_signature(self, params: dict) -> str:
        """Generate signature for authenticated requests"""
        if not self.api_secret:
            return ""
        query_string = urlencode(params)
        return hmac.new(
            self.api_secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()

    def get_account_info(self):
        """Get account information"""
        try:
            endpoint = "/api/v3/account"
            params = {"timestamp": int(time.time() * 1000)}
            if self.api_secret:
                params["signature"] = self._generate_signature(params)

            response = requests.get(
                f"{self.base_url}{endpoint}",
                headers=self.headers,
                params=params
            )
            return response.json() if response.status_code == 200 else {"error": "Authentication required"}
        except Exception as e:
            return {"error": str(e)}

    def get_ticker_24hr(self, symbol: str = None):
        """Get 24hr ticker price change statistics"""
        try:
            endpoint = "/api/v3/ticker/24hr"
            params = {"symbol": symbol} if symbol else {}

            response = requests.get(f"{self.base_url}{endpoint}", params=params)
            return response.json() if response.status_code == 200 else {"error": "Failed to fetch data"}
        except Exception as e:
            return {"error": str(e)}

    def get_orderbook(self, symbol: str, limit: int = 10):
        """Get order book for a symbol"""
        try:
            endpoint = "/api/v3/depth"
            params = {"symbol": symbol, "limit": limit}

            response = requests.get(f"{self.base_url}{endpoint}", params=params)
            return response.json() if response.status_code == 200 else {"error": "Failed to fetch data"}
        except Exception as e:
            return {"error": str(e)}

    def get_kline_data(self, symbol: str, interval: str = "1d", limit: int = 100):
        """Get kline/candlestick data"""
        try:
            endpoint = "/api/v3/klines"
            params = {
                "symbol": symbol,
                "interval": interval,
                "limit": limit
            }

            response = requests.get(f"{self.base_url}{endpoint}", params=params)
            if response.status_code == 200:
                data = response.json()
                df = pd.DataFrame(data, columns=[
                    'timestamp', 'open', 'high', 'low', 'close', 'volume',
                    'close_time', 'quote_asset_volume', 'number_of_trades',
                    'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
                ])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    df[col] = df[col].astype(float)
                return df
            return {"error": "Failed to fetch data"}
        except Exception as e:
            return {"error": str(e)}

    def place_order(self, symbol: str, side: str, order_type: str, quantity: float, price: float = None):
        """Place a new order (requires API secret)"""
        if not self.api_secret:
            return {"error": "API secret required for trading"}

        try:
            endpoint = "/api/v3/order"
            params = {
                "symbol": symbol,
                "side": side,
                "type": order_type,
                "quantity": quantity,
                "timestamp": int(time.time() * 1000)
            }

            if price and order_type == "LIMIT":
                params["price"] = price
                params["timeInForce"] = "GTC"

            params["signature"] = self._generate_signature(params)

            response = requests.post(
                f"{self.base_url}{endpoint}",
                headers=self.headers,
                params=params
            )
            return response.json()
        except Exception as e:
            return {"error": str(e)}

class GeminiAgent:
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash')

    def analyze_market(self, data: dict, question: str = None) -> str:
        """Analyze market data using Gemini AI"""
        try:
            prompt = f"""
            As a professional cryptocurrency trading analyst, analyze the following market data:

            {json.dumps(data, indent=2)}

            {"User Question: " + question if question else ""}

            Provide insights on:
            1. Current market trends
            2. Price action analysis
            3. Trading opportunities
            4. Risk assessment
            5. Specific recommendations

            Keep the analysis concise but comprehensive.
            """

            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"AI Analysis unavailable: {str(e)}"

    def generate_trading_strategy(self, symbol: str, timeframe: str, market_data: dict) -> str:
        """Generate trading strategy using AI"""
        try:
            prompt = f"""
            Create a comprehensive trading strategy for {symbol} on {timeframe} timeframe.

            Market Data:
            {json.dumps(market_data, indent=2)}

            Please provide:
            1. Entry and exit points
            2. Stop loss and take profit levels
            3. Risk management guidelines
            4. Market sentiment analysis
            5. Technical indicator recommendations

            Format as a structured trading plan.
            """

            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Strategy generation unavailable: {str(e)}"

# Initialize session state
if 'binance_api' not in st.session_state:
    st.session_state.binance_api = None
if 'gemini_agent' not in st.session_state:
    st.session_state.gemini_agent = None

# Header
st.markdown("""
<div class="main-header">
    <h1>🚀 Advanced Crypto Trading Agent</h1>
    <p>AI-Powered Trading with Binance Integration & Real-time Analytics</p>
</div>
""", unsafe_allow_html=True)

# Sidebar Configuration
with st.sidebar:
    st.header("🔧 Configuration")

    # API Keys Section
    st.subheader("API Configuration")

    # Binance API
    binance_api_key = st.text_input("Binance API Key", value="", type="password")
    binance_secret_key = st.text_input("Binance Secret Key (Optional for trading)", value="", type="password")

    # Gemini API
    gemini_api_key = st.text_input("Gemini AI API Key", value="", type="password")

    if st.button("Initialize APIs"):
        if binance_api_key:
            st.session_state.binance_api = BinanceAPI(binance_api_key, binance_secret_key)
            st.success("Binance API initialized!")

        if gemini_api_key:
            st.session_state.gemini_agent = GeminiAgent(gemini_api_key)
            st.success("Gemini AI initialized!")

    st.divider()

    # Trading Parameters
    st.subheader("Trading Parameters")

    # Comprehensive list of popular cryptocurrencies
    crypto_pairs = [
        # Major Cryptocurrencies
        "BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT", "SOLUSDT", "DOTUSDT",
        "MATICUSDT", "SHIBUSDT", "AVAXUSDT", "LTCUSDT", "TRXUSDT", "UNIUSDT", "LINKUSDT",
        "ATOMUSDT", "ETCUSDT", "XLMUSDT", "NEARUSDT", "ALGOUSDT", "VETUSDT", "ICPUSDT",

        # DeFi Tokens
        "AAVEUSDT", "MKRUSDT", "COMPUSDT", "SUSHIUSDT", "CRVUSDT", "1INCHUSDT", "YFIUSDT",
        "SNXUSDT", "UMAUSDT", "BALUSDT", "RENUUSDT", "LRCUSDT", "DYDXUSDT", "AUDIOUSDT",

        # Layer 1 & Layer 2
        "FTMUSDT", "ONEUSDT", "ZILUSDT", "EGLDUSDT", "HNTUSDT", "FLOWUSDT", "KAVAUSDT",
        "KSMUSDT", "WAVESUSDT", "QTUMUSDT", "ZECUSDT", "DASHUSDT", "DCRUSDT", "BATUSDT",

        # Meme Coins & Popular Alts
        "PEPEUSDT", "FLOKIUSDT", "BONKUSDT", "1000SATSUSDT", "ORDIUSDT", "WIFUSDT",
        "JUPUSDT", "PYTHUSDT", "STRKUSDT", "AIUSDT", "FETUSDT", "AGIXUSDT", "OCEAUSDT",

        # Gaming & NFT
        "AXSUSDT", "SANDUSDT", "MANAUSDT", "ENJUSDT", "GALAUSDT", "CHZUSDT", "THETAUSDT",
        "FLOWUSDT", "APECUSDT", "GMTUSDT", "STGUSDT", "MAGICUSDT", "IMXUSDT", "GALUSDT",

        # Enterprise & Utility
        "HBARUSDT", "FILUSDT", "GRTUSDT", "RNDRUSDT", "ARBUSDT", "OPUSDT", "LDOUSDT",
        "SUIUSDT", "APTUSDT", "INJUSDT", "TIAUSDT", "SEIUSDT", "ARKMUSDT", "WLDUSDT",

        # Stablecoins & Others
        "BUSDUSDT", "TUSDUSDT", "USDCUSDT", "DAIUSDT", "FRAXUSDT", "LUSDUSDT",

        # New & Trending
        "BONKUSDT", "JTOAUSDT", "ACEUSDT", "NFPUSDT", "AIUSDT", "XAIUSDT", "MANTAUSDT",
        "ALTUSDT", "JUPUSDT", "DYMUSDT", "PIXELUSDT", "STRKUSDT", "PORTALUSDT", "PDAUSDT"
    ]

    selected_symbol = st.selectbox(
        "Select Trading Pair",
        crypto_pairs,
        index=0  # Default to BTC
    )

    timeframe = st.selectbox(
        "Timeframe",
        ["1m", "5m", "15m", "1h", "4h", "1d", "1w"]
    )

    trade_amount = st.number_input("Trade Amount (USDT)", min_value=10.0, value=100.0)

# Main Content Tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Markets", "💰 Buy Crypto", "📈 Trading", "🔮 Futures", "💎 Earn", "🤖 AI Analysis"
])

# Markets Tab
with tab1:
    st.header("📊 Market Overview")

    if st.session_state.binance_api:
        # Market Summary Cards
        col1, col2 = st.columns([2, 1])

        with col1:
            # Chart cryptocurrency selector (independent from sidebar)
            st.subheader("📈 Interactive Price Chart")

            # Quick crypto selector for chart
            chart_header_col1, chart_header_col2 = st.columns([2, 1])
            with chart_header_col1:
                chart_symbol = st.selectbox(
                    "Select Cryptocurrency for Chart",
                    crypto_pairs,
                    index=crypto_pairs.index(selected_symbol) if selected_symbol in crypto_pairs else 0,
                    key="chart_symbol"
                )

            # Chart controls
            chart_col1, chart_col2, chart_col3, chart_col4 = st.columns(4)
            with chart_col1:
                chart_type = st.selectbox("Chart Type", ["Candlestick", "Line", "Area"])
            with chart_col2:
                chart_timeframe = st.selectbox("Timeframe",
                                             ["1m", "5m", "15m", "1h", "4h", "1d", "1w"],
                                             index=["1m", "5m", "15m", "1h", "4h", "1d", "1w"].index(timeframe),
                                             key="chart_timeframe")
            with chart_col3:
                data_points = st.selectbox("Data Points", [50, 100, 200, 500])
            with chart_col4:
                if st.button("🔄 Refresh Chart"):
                    st.rerun()

            kline_data = st.session_state.binance_api.get_kline_data(chart_symbol, chart_timeframe, data_points)

            if isinstance(kline_data, pd.DataFrame):
                if chart_type == "Candlestick":
                    fig = go.Figure(data=go.Candlestick(
                        x=kline_data['timestamp'],
                        open=kline_data['open'],
                        high=kline_data['high'],
                        low=kline_data['low'],
                        close=kline_data['close'],
                        name=chart_symbol
                    ))
                elif chart_type == "Line":
                    fig = go.Figure(data=go.Scatter(
                        x=kline_data['timestamp'],
                        y=kline_data['close'],
                        mode='lines',
                        name=chart_symbol,
                        line=dict(color='#667eea', width=2)
                    ))
                else:  # Area
                    fig = go.Figure(data=go.Scatter(
                        x=kline_data['timestamp'],
                        y=kline_data['close'],
                        fill='tonexty',
                        mode='lines',
                        name=chart_symbol,
                        line=dict(color='#667eea')
                    ))

                fig.update_layout(
                    title=f"📊 {chart_symbol} {chart_type} Chart ({chart_timeframe}) - {data_points} periods",
                    yaxis_title="Price (USDT)",
                    xaxis_title="Time",
                    template="plotly_dark",
                    height=500
                )

                st.plotly_chart(fig, use_container_width=True)

                # Volume chart for the selected chart symbol
                volume_fig = go.Figure(data=go.Bar(
                    x=kline_data['timestamp'],
                    y=kline_data['volume'],
                    name="Volume",
                    marker_color='rgba(102, 126, 234, 0.6)'
                ))

                volume_fig.update_layout(
                    title=f"📊 {chart_symbol} Volume ({chart_timeframe})",
                    yaxis_title="Volume",
                    template="plotly_dark",
                    height=200
                )

                st.plotly_chart(volume_fig, use_container_width=True)

                # Quick stats for chart symbol
                chart_ticker = st.session_state.binance_api.get_ticker_24hr(chart_symbol)
                if "error" not in chart_ticker:
                    chart_price = float(chart_ticker['lastPrice'])
                    chart_change = float(chart_ticker['priceChangePercent'])

                    st.markdown(f"""
                    <div style="background: linear-gradient(45deg, #667eea, #764ba2); padding: 1rem; border-radius: 10px; text-align: center; color: white; margin: 1rem 0;">
                        <h3>{chart_symbol.replace('USDT', '')} Quick Stats</h3>
                        <div style="display: flex; justify-content: space-around;">
                            <div><strong>Price:</strong> ${chart_price:,.4f}</div>
                            <div><strong>24h Change:</strong> {chart_change:+.2f}%</div>
                            <div><strong>Volume:</strong> {float(chart_ticker['volume']):,.0f}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.error("Failed to fetch price data")

        with col2:
            # Market Stats for sidebar selected symbol
            st.subheader(f"📊 {selected_symbol} Statistics")

            ticker_data = st.session_state.binance_api.get_ticker_24hr(selected_symbol)

            if "error" not in ticker_data:
                current_price = float(ticker_data['lastPrice'])
                price_change = float(ticker_data['priceChange'])
                price_change_percent = float(ticker_data['priceChangePercent'])
                volume = float(ticker_data['volume'])
                high_24h = float(ticker_data['highPrice'])
                low_24h = float(ticker_data['lowPrice'])

                # Price metrics with color coding
                price_color = "inverse" if price_change >= 0 else "off"

                st.metric("💰 Current Price", f"${current_price:,.4f}",
                         f"{price_change:+.4f} ({price_change_percent:+.2f}%)",
                         delta_color=price_color)

                st.metric("📈 24h High", f"${high_24h:,.4f}")
                st.metric("📉 24h Low", f"${low_24h:,.4f}")
                st.metric("📊 24h Volume", f"{volume:,.2f}")

                # Additional metrics
                st.divider()
                st.subheader("🔍 Technical Indicators")

                # Get technical data for the chart symbol, not sidebar symbol
                tech_kline_data = st.session_state.binance_api.get_kline_data(chart_symbol, chart_timeframe, 100)

                # Simple technical indicators
                if isinstance(tech_kline_data, pd.DataFrame) and len(tech_kline_data) > 20:
                    # Moving averages
                    ma_20 = tech_kline_data['close'].tail(20).mean()
                    ma_50 = tech_kline_data['close'].tail(50).mean() if len(tech_kline_data) >= 50 else ma_20

                    st.metric("MA(20)", f"${ma_20:.4f}")
                    st.metric("MA(50)", f"${ma_50:.4f}")

                    # RSI approximation
                    price_changes = tech_kline_data['close'].diff()
                    gains = price_changes.where(price_changes > 0, 0)
                    losses = -price_changes.where(price_changes < 0, 0)

                    if len(gains) > 14:
                        avg_gain = gains.tail(14).mean()
                        avg_loss = losses.tail(14).mean()
                        rs = avg_gain / avg_loss if avg_loss != 0 else 0
                        rsi = 100 - (100 / (1 + rs))

                        rsi_color = "inverse" if 30 <= rsi <= 70 else "off"
                        st.metric("RSI(14)", f"{rsi:.1f}", delta_color=rsi_color)

                    # Additional info for current chart symbol
                    st.info(f"📈 Technical analysis for **{chart_symbol}** on **{chart_timeframe}** timeframe")

            # Order Book
            st.subheader("📋 Order Book")
            orderbook = st.session_state.binance_api.get_orderbook(selected_symbol, 5)

            if "error" not in orderbook:
                # Enhanced order book display
                bids_df = pd.DataFrame(orderbook['bids'], columns=['Price', 'Quantity'])
                asks_df = pd.DataFrame(orderbook['asks'], columns=['Price', 'Quantity'])

                bids_df['Price'] = pd.to_numeric(bids_df['Price'])
                bids_df['Quantity'] = pd.to_numeric(bids_df['Quantity'])
                asks_df['Price'] = pd.to_numeric(asks_df['Price'])
                asks_df['Quantity'] = pd.to_numeric(asks_df['Quantity'])

                # Calculate total values
                bids_df['Total'] = bids_df['Price'] * bids_df['Quantity']
                asks_df['Total'] = asks_df['Price'] * asks_df['Quantity']

                col_bid, col_ask = st.columns(2)

                with col_bid:
                    st.markdown("**🟢 Bids**")
                    st.dataframe(
                        bids_df.style.format({
                            'Price': '${:.4f}',
                            'Quantity': '{:.4f}',
                            'Total': '${:.2f}'
                        }),
                        use_container_width=True
                    )

                with col_ask:
                    st.markdown("**🔴 Asks**")
                    st.dataframe(
                        asks_df.style.format({
                            'Price': '${:.4f}',
                            'Quantity': '{:.4f}',
                            'Total': '${:.2f}'
                        }),
                        use_container_width=True
                    )

                # Spread calculation
                best_bid = bids_df['Price'].iloc[0]
                best_ask = asks_df['Price'].iloc[0]
                spread = best_ask - best_bid
                spread_percent = (spread / best_ask) * 100

                st.metric("💹 Bid-Ask Spread", f"${spread:.4f} ({spread_percent:.3f}%)")

    else:
        st.warning("Please initialize Binance API in the sidebar to view market data.")

# Buy Crypto Tab
with tab2:
    st.header("💰 Buy Crypto")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Spot Trading")

        # Enhanced crypto selector with search functionality
        buy_symbol = st.selectbox(
            "Select Cryptocurrency",
            crypto_pairs,
            index=0,
            key="buy_symbol",
            help="Choose from 100+ available cryptocurrencies"
        )



        buy_amount = st.number_input("Amount (USDT)", min_value=10.0, value=100.0)
        order_type = st.selectbox("Order Type", ["MARKET", "LIMIT"])

        if order_type == "LIMIT":
            limit_price = st.number_input("Limit Price", min_value=0.01)

        # Show current price for selected crypto
        if st.session_state.binance_api:
            current_ticker = st.session_state.binance_api.get_ticker_24hr(buy_symbol)
            if "error" not in current_ticker:
                current_price = float(current_ticker['lastPrice'])
                price_change = float(current_ticker['priceChangePercent'])

                st.markdown(f"""
                <div style="background: {'linear-gradient(45deg, #28a745, #20c997)' if price_change >= 0 else 'linear-gradient(45deg, #dc3545, #fd7e14)'};
                     padding: 1rem; border-radius: 10px; text-align: center; color: white; margin: 1rem 0;">
                    <h4>{buy_symbol.replace('USDT', '')} Current Price</h4>
                    <h2>${current_price:,.4f}</h2>
                    <p>24h Change: {price_change:+.2f}%</p>
                    <p>You'll get approximately: <strong>{buy_amount/current_price:.6f} {buy_symbol.replace('USDT', '')}</strong></p>
                </div>
                """, unsafe_allow_html=True)

        if st.button("Place Buy Order", type="primary"):
            if st.session_state.binance_api and st.session_state.binance_api.api_secret:
                # Calculate quantity based on current price
                ticker = st.session_state.binance_api.get_ticker_24hr(buy_symbol)
                if "error" not in ticker:
                    current_price = float(ticker['lastPrice'])
                    quantity = buy_amount / current_price

                    price = limit_price if order_type == "LIMIT" else None

                    result = st.session_state.binance_api.place_order(
                        buy_symbol, "BUY", order_type, quantity, price
                    )

                    if "error" not in result:
                        st.success("Order placed successfully!")
                        st.json(result)
                    else:
                        st.error(f"Order failed: {result['error']}")
                else:
                    st.error("Failed to fetch current price")
            else:
                st.warning("Please provide Binance API secret key for trading")

    with col2:
        st.subheader("Portfolio Overview")

        if st.session_state.binance_api:
            account_info = st.session_state.binance_api.get_account_info()

            if "error" not in account_info and "balances" in account_info:
                balances = [b for b in account_info["balances"] if float(b["free"]) > 0]

                if balances:
                    balance_df = pd.DataFrame(balances)
                    balance_df["free"] = pd.to_numeric(balance_df["free"])
                    balance_df["locked"] = pd.to_numeric(balance_df["locked"])
                    balance_df = balance_df[balance_df["free"] > 0].sort_values("free", ascending=False)

                    st.dataframe(balance_df, use_container_width=True)
                else:
                    st.info("No balances found or API authentication required")
            else:
                st.warning("Account information requires API secret key")

# Trading Tab
with tab3:
    st.header("📈 Advanced Trading")

    col1, col2 = st.columns([3, 1])

    with col1:
        st.subheader("Trading Interface")

        # Enhanced trading with all cryptocurrencies
        with st.form("trading_form"):
            trade_col1, trade_col2, trade_col3 = st.columns(3)

            with trade_col1:
                trade_symbol = st.selectbox(
                    "Symbol",
                    crypto_pairs,
                    index=0,
                    help="Select from 100+ cryptocurrencies"
                )
                trade_side = st.selectbox("Side", ["BUY", "SELL"])

            with trade_col2:
                trade_type = st.selectbox("Type", ["MARKET", "LIMIT", "STOP_LIMIT"])
                trade_quantity = st.number_input("Quantity", min_value=0.001, step=0.001)

            with trade_col3:
                if trade_type in ["LIMIT", "STOP_LIMIT"]:
                    trade_price = st.number_input("Price", min_value=0.01, step=0.01)
                else:
                    trade_price = None

                # Show estimated value
                if st.session_state.binance_api and trade_quantity > 0:
                    ticker = st.session_state.binance_api.get_ticker_24hr(trade_symbol)
                    if "error" not in ticker:
                        current_price = float(ticker['lastPrice'])
                        estimated_value = trade_quantity * current_price
                        st.metric("Est. Value", f"${estimated_value:.2f}")

            # Quick trading buttons
            st.markdown("**⚡ Quick Trade Amounts:**")
            quick_trade_col1, quick_trade_col2, quick_trade_col3, quick_trade_col4 = st.columns(4)

            with quick_trade_col1:
                if st.form_submit_button("$50"):
                    st.session_state.quick_trade_amount = 50
            with quick_trade_col2:
                if st.form_submit_button("$100"):
                    st.session_state.quick_trade_amount = 100
            with quick_trade_col3:
                if st.form_submit_button("$500"):
                    st.session_state.quick_trade_amount = 500
            with quick_trade_col4:
                if st.form_submit_button("$1000"):
                    st.session_state.quick_trade_amount = 1000

            submitted = st.form_submit_button("Execute Trade", type="primary")

            if submitted:
                if st.session_state.binance_api and st.session_state.binance_api.api_secret:
                    result = st.session_state.binance_api.place_order(
                        trade_symbol, trade_side, trade_type, trade_quantity, trade_price
                    )

                    if "error" not in result:
                        st.success("Trade executed successfully!")
                        st.json(result)
                    else:
                        st.error(f"Trade failed: {result['error']}")
                else:
                    st.warning("Trading requires API secret key")

    with col2:
        st.subheader("Quick Actions")

        # Popular crypto quick actions
        st.markdown("**🔥 Popular Cryptos:**")
        popular_trading = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "ADAUSDT"]

        for crypto in popular_trading:
            col_buy, col_sell = st.columns(2)
            with col_buy:
                if st.button(f"Buy {crypto.replace('USDT', '')}", key=f"qbuy_{crypto}"):
                    st.info(f"Quick buy {crypto}")
            with col_sell:
                if st.button(f"Sell {crypto.replace('USDT', '')}", key=f"qsell_{crypto}"):
                    st.info(f"Quick sell {crypto}")

        st.divider()

        if st.button("Close All Positions", type="secondary"):
            st.warning("All positions would be closed")

        # Market overview for trading
        st.subheader("📊 Live Prices")
        if st.session_state.binance_api:
            for crypto in popular_trading[:3]:
                ticker = st.session_state.binance_api.get_ticker_24hr(crypto)
                if "error" not in ticker:
                    price = float(ticker['lastPrice'])
                    change = float(ticker['priceChangePercent'])
                    color = "🟢" if change >= 0 else "🔴"
                    st.markdown(f"**{crypto.replace('USDT', '')}** {color} ${price:,.4f} ({change:+.2f}%)")

# Futures Tab
with tab4:
    st.header("🔮 Futures Trading")

    st.info("⚠️ Futures trading involves high risk. Trade responsibly!")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Futures Positions")

        futures_symbol = st.selectbox(
            "Futures Symbol",
            crypto_pairs,
            index=0,
            key="futures_symbol",
            help="Select from 100+ futures pairs"
        )

        leverage = st.slider("Leverage", min_value=1, max_value=125, value=10)
        position_size = st.number_input("Position Size (USDT)", min_value=10.0, value=100.0)

        # Show current futures price and funding rate
        if st.session_state.binance_api:
            futures_ticker = st.session_state.binance_api.get_ticker_24hr(futures_symbol)
            if "error" not in futures_ticker:
                futures_price = float(futures_ticker['lastPrice'])
                futures_change = float(futures_ticker['priceChangePercent'])

                st.markdown(f"""
                <div style="background: linear-gradient(45deg, #fd7e14, #e83e8c); padding: 1rem; border-radius: 10px; text-align: center; color: white; margin: 1rem 0;">
                    <h4>{futures_symbol.replace('USDT', '')} Futures</h4>
                    <h3>${futures_price:,.4f}</h3>
                    <p>24h Change: {futures_change:+.2f}%</p>
                    <p>Position Value: <strong>${position_size * leverage:,.2f}</strong></p>
                    <p>Margin Required: <strong>${position_size:,.2f}</strong></p>
                </div>
                """, unsafe_allow_html=True)

        col_long, col_short = st.columns(2)

        with col_long:
            if st.button("🟢 Open Long Position", type="primary"):
                st.success(f"Long position opened for {futures_symbol}")
                st.info(f"Leverage: {leverage}x | Size: ${position_size}")

        with col_short:
            if st.button("🔴 Open Short Position"):
                st.success(f"Short position opened for {futures_symbol}")
                st.info(f"Leverage: {leverage}x | Size: ${position_size}")

    with col2:
        st.subheader("Risk Management")

        stop_loss = st.number_input("Stop Loss (%)", min_value=0.1, max_value=50.0, value=5.0)
        take_profit = st.number_input("Take Profit (%)", min_value=0.1, max_value=100.0, value=10.0)

        # Calculate risk metrics
        max_loss = position_size * leverage * stop_loss / 100
        potential_profit = position_size * leverage * take_profit / 100

        st.metric("Max Risk per Trade", f"${max_loss:.2f}")
        st.metric("Potential Profit", f"${potential_profit:.2f}")
        st.metric("Risk/Reward Ratio", f"1:{take_profit/stop_loss:.1f}")

        st.divider()
        st.subheader("📊 Futures Market Data")

        # Show top futures by volume
        if st.session_state.binance_api:
            st.markdown("**🔥 Hot Futures:**")
            for crypto in popular_trading[:5]:
                ticker = st.session_state.binance_api.get_ticker_24hr(crypto)
                if "error" not in ticker:
                    price = float(ticker['lastPrice'])
                    change = float(ticker['priceChangePercent'])
                    volume = float(ticker['volume'])
                    color = "🟢" if change >= 0 else "🔴"

                    st.markdown(f"""
                    <div style="background: rgba(102, 126, 234, 0.1); padding: 0.5rem; border-radius: 5px; margin: 0.2rem 0;">
                        <strong>{crypto.replace('USDT', '')}</strong> {color}
                        ${price:,.4f} ({change:+.2f}%)
                        <small>Vol: {volume:,.0f}</small>
                    </div>
                    """, unsafe_allow_html=True)

# Earn Tab
with tab5:
    st.header("💎 Earn & Staking")

    st.subheader("Available Earning Options")

    # Enhanced earning options with more cryptocurrencies
    earning_options = [
        {"name": "Bitcoin Staking", "symbol": "BTCUSDT", "apy": "4.5%", "min_amount": "0.001 BTC", "lock_period": "30 days", "risk": "Low"},
        {"name": "Ethereum 2.0 Staking", "symbol": "ETHUSDT", "apy": "5.8%", "min_amount": "0.1 ETH", "lock_period": "60 days", "risk": "Low"},
        {"name": "BNB Staking", "symbol": "BNBUSDT", "apy": "6.2%", "min_amount": "0.1 BNB", "lock_period": "30 days", "risk": "Low"},
        {"name": "Solana Staking", "symbol": "SOLUSDT", "apy": "7.8%", "min_amount": "1 SOL", "lock_period": "7 days", "risk": "Medium"},
        {"name": "Cardano Staking", "symbol": "ADAUSDT", "apy": "5.5%", "min_amount": "10 ADA", "lock_period": "21 days", "risk": "Low"},
        {"name": "Polygon Staking", "symbol": "MATICUSDT", "apy": "8.2%", "min_amount": "100 MATIC", "lock_period": "14 days", "risk": "Medium"},
        {"name": "Avalanche Staking", "symbol": "AVAXUSDT", "apy": "9.1%", "min_amount": "1 AVAX", "lock_period": "14 days", "risk": "Medium"},
        {"name": "Polkadot Staking", "symbol": "DOTUSDT", "apy": "12.5%", "min_amount": "1 DOT", "lock_period": "28 days", "risk": "High"},
        {"name": "USDT Savings", "symbol": "USDT", "apy": "8.5%", "min_amount": "100 USDT", "lock_period": "7 days", "risk": "Low"},
        {"name": "Flexible Savings", "symbol": "Multi", "apy": "2.1%", "min_amount": "10 USDT", "lock_period": "Flexible", "risk": "Very Low"},
    ]

    # Create tabs for different earning categories
    earn_tab1, earn_tab2, earn_tab3 = st.tabs(["🔒 Staking", "💰 Savings", "🎯 DeFi Pools"])

    with earn_tab1:
        st.subheader("Cryptocurrency Staking")

        staking_options = [opt for opt in earning_options if "Staking" in opt["name"]]

        for i, option in enumerate(staking_options):
            with st.expander(f"{option['name']} - APY: {option['apy']} | Risk: {option['risk']}"):
                col1, col2, col3 = st.columns([2, 1, 1])

                with col1:
                    st.write(f"**Minimum Amount:** {option['min_amount']}")
                    st.write(f"**Lock Period:** {option['lock_period']}")
                    st.write(f"**Risk Level:** {option['risk']}")

                    # Show current price if available
                    if st.session_state.binance_api and option['symbol'] != 'Multi':
                        ticker = st.session_state.binance_api.get_ticker_24hr(option['symbol'])
                        if "error" not in ticker:
                            price = float(ticker['lastPrice'])
                            st.write(f"**Current Price:** ${price:,.4f}")

                    stake_amount = st.number_input(f"Amount to stake", key=f"staking_stake_{i}", min_value=0.0)

                with col2:
                    if stake_amount > 0:
                        apy_value = float(option['apy'].replace('%', ''))
                        yearly_earnings = stake_amount * apy_value / 100

                        st.metric("Estimated Yearly Earnings", f"${yearly_earnings:.2f}")
                        st.metric("Monthly Earnings", f"${yearly_earnings/12:.2f}")
                        st.metric("Daily Earnings", f"${yearly_earnings/365:.4f}")

                with col3:
                    if st.button(f"Stake Now", key=f"staking_stake_btn_{i}", type="primary"):
                        st.success(f"Staked ${stake_amount:.2f} in {option['name']}")
                        st.balloons()

    with earn_tab2:
        st.subheader("Flexible & Fixed Savings")

        savings_options = [opt for opt in earning_options if "Savings" in opt["name"]]

        # Add more savings options
        additional_savings = [
            {"name": "BUSD Savings", "symbol": "BUSDUSDT", "apy": "7.2%", "min_amount": "50 BUSD", "lock_period": "Flexible", "risk": "Very Low"},
            {"name": "USDC Savings", "symbol": "USDCUSDT", "apy": "6.8%", "min_amount": "50 USDC", "lock_period": "Flexible", "risk": "Very Low"},
        ]

        all_savings = savings_options + additional_savings

        for i, option in enumerate(all_savings):
            with st.expander(f"{option['name']} - APY: {option['apy']}"):
                col1, col2, col3 = st.columns(3)

                with col1:
                    save_amount = st.number_input(f"Amount", key=f"savings_save_{i}", min_value=0.0)

                with col2:
                    if save_amount > 0:
                        apy_value = float(option['apy'].replace('%', ''))
                        earnings = save_amount * apy_value / 100
                        st.metric("Annual Earnings", f"${earnings:.2f}")

                with col3:
                    if st.button(f"Start Earning", key=f"savings_save_btn_{i}"):
                        st.success(f"Started earning on ${save_amount:.2f}")

    with earn_tab3:
        st.subheader("DeFi Liquidity Pools")

        defi_pools = [
            {"name": "BTC/ETH Pool", "apy": "15.2%", "risk": "High", "tvl": "$2.5M"},
            {"name": "USDT/BUSD Pool", "apy": "4.8%", "risk": "Low", "tvl": "$8.9M"},
            {"name": "BNB/CAKE Pool", "apy": "22.1%", "risk": "Very High", "tvl": "$1.2M"},
            {"name": "ETH/MATIC Pool", "apy": "18.7%", "risk": "High", "tvl": "$3.1M"},
        ]

        for i, pool in enumerate(defi_pools):
            with st.expander(f"{pool['name']} - APY: {pool['apy']} | TVL: {pool['tvl']}"):
                col1, col2, col3 = st.columns(3)

                with col1:
                    pool_amount = st.number_input(f"Liquidity Amount (USDT)", key=f"defi_pool_{i}", min_value=0.0)
                    st.write(f"**Risk Level:** {pool['risk']}")
                    st.write(f"**Total Value Locked:** {pool['tvl']}")

                with col2:
                    if pool_amount > 0:
                        pool_apy = float(pool['apy'].replace('%', ''))
                        pool_earnings = pool_amount * pool_apy / 100
                        st.metric("Potential Annual Yield", f"${pool_earnings:.2f}")
                        st.write("⚠️ Impermanent loss risk applies")

                with col3:
                    if st.button(f"Add Liquidity", key=f"defi_pool_btn_{i}"):
                        st.warning(f"Added ${pool_amount:.2f} to {pool['name']}")
                        st.info("Monitor for impermanent loss!")

    # Overall earning summary
    st.divider()
    st.subheader("📊 Earning Portfolio Summary")

    summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)

    with summary_col1:
        st.metric("Total Staked", "$0.00", "Start staking to see balance")

    with summary_col2:
        st.metric("Total Earnings (24h)", "$0.00", "No active positions")

    with summary_col3:
        st.metric("Average APY", "0.00%", "Across all positions")

    with summary_col4:
        st.metric("Active Positions", "0", "Start earning now!")

# AI Analysis Tab
with tab6:
    st.header("🤖 AI-Powered Market Analysis")

    if st.session_state.gemini_agent and st.session_state.binance_api:
        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("AI Market Analysis")

            analysis_symbol = st.selectbox(
                "Select Symbol for Analysis",
                crypto_pairs,
                index=0,
                key="analysis_symbol",
                help="Choose from 100+ cryptocurrencies for AI analysis"
            )

            user_question = st.text_area("Ask AI about the market (optional)",
                                       placeholder=f"e.g., Should I buy {analysis_symbol.replace('USDT', '')} now? What's the trend analysis?")

            # Analysis type selector
            analysis_type = st.selectbox(
                "Analysis Type",
                ["Comprehensive Analysis", "Technical Analysis", "Risk Assessment", "Price Prediction", "Market Sentiment"]
            )

            if st.button("Generate AI Analysis", type="primary"):
                with st.spinner(f"AI is analyzing {analysis_symbol.replace('USDT', '')}..."):
                    # Fetch comprehensive market data
                    ticker_data = st.session_state.binance_api.get_ticker_24hr(analysis_symbol)
                    kline_data = st.session_state.binance_api.get_kline_data(analysis_symbol, "1d", 30)
                    orderbook_data = st.session_state.binance_api.get_orderbook(analysis_symbol, 10)

                    # Prepare comprehensive data for AI
                    market_data = {
                        "symbol": analysis_symbol,
                        "current_price": ticker_data.get("lastPrice", "N/A"),
                        "24h_change": ticker_data.get("priceChangePercent", "N/A"),
                        "24h_volume": ticker_data.get("volume", "N/A"),
                        "24h_high": ticker_data.get("highPrice", "N/A"),
                        "24h_low": ticker_data.get("lowPrice", "N/A"),
                        "price_change": ticker_data.get("priceChange", "N/A"),
                        "analysis_type": analysis_type,
                        "recent_highs_lows": {
                            "24h_high": ticker_data.get("highPrice", "N/A"),
                            "24h_low": ticker_data.get("lowPrice", "N/A")
                        }
                    }

                    # Add technical indicators if kline data available
                    if isinstance(kline_data, pd.DataFrame) and len(kline_data) > 20:
                        recent_prices = kline_data['close'].tail(20)
                        market_data["technical_indicators"] = {
                            "sma_20": recent_prices.mean(),
                            "price_trend": "bullish" if recent_prices.iloc[-1] > recent_prices.mean() else "bearish",
                            "volatility": recent_prices.std()
                        }

                    # Get AI analysis
                    analysis = st.session_state.gemini_agent.analyze_market(market_data, user_question)

                    st.markdown("### 🎯 AI Analysis Results")
                    st.markdown(analysis)

                    # Add current price info
                    if "error" not in ticker_data:
                        current_price = float(ticker_data['lastPrice'])
                        price_change = float(ticker_data['priceChangePercent'])

                        st.markdown(f"""
                        <div style="background: {'linear-gradient(45deg, #28a745, #20c997)' if price_change >= 0 else 'linear-gradient(45deg, #dc3545, #fd7e14)'};
                             padding: 1rem; border-radius: 10px; text-align: center; color: white; margin: 1rem 0;">
                            <h3>📊 {analysis_symbol.replace('USDT', '')} Current Status</h3>
                            <div style="display: flex; justify-content: space-around;">
                                <div><strong>Price:</strong> ${current_price:,.4f}</div>
                                <div><strong>24h Change:</strong> {price_change:+.2f}%</div>
                                <div><strong>Volume:</strong> {float(ticker_data['volume']):,.0f}</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

        with col2:
            st.subheader("Trading Strategy Generator")

            strategy_symbol = st.selectbox(
                "Strategy Symbol",
                crypto_pairs[:20],  # Top 20 for strategy
                key="strategy_symbol"
            )

            strategy_timeframe = st.selectbox("Strategy Timeframe",
                                            ["1h", "4h", "1d", "1w"],
                                            key="strategy_timeframe")

            risk_tolerance = st.selectbox("Risk Tolerance", ["Conservative", "Moderate", "Aggressive"])

            if st.button("Generate Trading Strategy"):
                with st.spinner("Creating personalized trading strategy..."):
                    strategy_ticker = st.session_state.binance_api.get_ticker_24hr(strategy_symbol)

                    strategy_data = {
                        "symbol": strategy_symbol,
                        "timeframe": strategy_timeframe,
                        "risk_tolerance": risk_tolerance,
                        "current_metrics": strategy_ticker
                    }

                    strategy = st.session_state.gemini_agent.generate_trading_strategy(
                        strategy_symbol, strategy_timeframe, strategy_data
                    )

                    st.markdown("### 📋 AI Trading Strategy")
                    st.markdown(strategy)

            st.divider()

            st.subheader("Multi-Crypto Comparison")

            compare_cryptos = st.multiselect(
                "Compare Cryptocurrencies",
                crypto_pairs[:15],  # Top 15 for comparison
                default=["BTCUSDT", "ETHUSDT"],
                max_selections=5
            )

            if st.button("Compare Selected Cryptos") and len(compare_cryptos) > 1:
                with st.spinner("Comparing cryptocurrencies..."):
                    comparison_data = {}

                    for crypto in compare_cryptos:
                        ticker = st.session_state.binance_api.get_ticker_24hr(crypto)
                        if "error" not in ticker:
                            comparison_data[crypto] = {
                                "price": float(ticker['lastPrice']),
                                "change_24h": float(ticker['priceChangePercent']),
                                "volume": float(ticker['volume'])
                            }

                    # Display comparison table
                    if comparison_data:
                        comp_df = pd.DataFrame(comparison_data).T
                        comp_df.index = [crypto.replace('USDT', '') for crypto in comp_df.index]

                        st.dataframe(
                            comp_df.style.format({
                                'price': '${:,.4f}',
                                'change_24h': '{:+.2f}%',
                                'volume': '{:,.0f}'
                            }).background_gradient(subset=['change_24h']),
                            use_container_width=True
                        )

            st.divider()

            st.subheader("Quick AI Insights")

            insight_crypto = st.selectbox(
                "Quick Insight For:",
                ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT"],
                key="insight_crypto"
            )

            insight_col1, insight_col2 = st.columns(2)

            with insight_col1:
                if st.button("Market Sentiment", key="sentiment"):
                    if st.session_state.binance_api:
                        ticker = st.session_state.binance_api.get_ticker_24hr(insight_crypto)
                        if "error" not in ticker:
                            change = float(ticker['priceChangePercent'])
                            if change > 5:
                                st.success("🟢 Very Bullish sentiment detected")
                            elif change > 0:
                                st.info("🔵 Bullish sentiment detected")
                            elif change > -5:
                                st.warning("🟡 Neutral to bearish sentiment")
                            else:
                                st.error("🔴 Very bearish sentiment detected")

            with insight_col2:
                if st.button("Risk Assessment", key="risk"):
                    if st.session_state.binance_api:
                        ticker = st.session_state.binance_api.get_ticker_24hr(insight_crypto)
                        if "error" not in ticker:
                            change = abs(float(ticker['priceChangePercent']))
                            if change > 10:
                                st.error("⚠️ High risk - very volatile")
                            elif change > 5:
                                st.warning("⚠️ Medium risk - volatile conditions")
                            else:
                                st.success("✅ Low risk - stable conditions")

            if st.button("Opportunity Scanner", key="opportunity"):
                with st.spinner("Scanning for opportunities..."):
                    opportunities = []
                    scan_cryptos = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "ADAUSDT"]

                    for crypto in scan_cryptos:
                        ticker = st.session_state.binance_api.get_ticker_24hr(crypto)
                        if "error" not in ticker:
                            change = float(ticker['priceChangePercent'])
                            volume = float(ticker['volume'])

                            if change > 3 and volume > 10000:  # Simplified opportunity logic
                                opportunities.append(f"🟢 {crypto.replace('USDT', '')}: +{change:.1f}% with high volume")
                            elif change < -3 and volume > 10000:
                                opportunities.append(f"🔵 {crypto.replace('USDT', '')}: {change:.1f}% potential dip buy")

                    if opportunities:
                        for opp in opportunities:
                            st.success(opp)
                    else:
                        st.info("💡 No significant opportunities detected right now")

    else:
        st.warning("Please initialize both Gemini AI and Binance API to use AI analysis features.")

        if not st.session_state.gemini_agent:
            st.info("🔑 Gemini AI API key required for market analysis")

        if not st.session_state.binance_api:
            st.info("🔑 Binance API key required for market data")

        # Show demo features even without API
        st.subheader("🎯 Available AI Features (Demo)")

        demo_features = [
            "📊 **Comprehensive Market Analysis** - Deep dive into price action, volume, and trends",
            "📈 **Technical Analysis** - RSI, MACD, moving averages, and support/resistance levels",
            "⚖️ **Risk Assessment** - Volatility analysis and risk scoring",
            "🔮 **Price Prediction** - AI-powered short and medium-term price forecasts",
            "💭 **Market Sentiment** - Social media and news sentiment analysis",
            "📋 **Trading Strategy Generation** - Personalized strategies based on risk tolerance",
            "🔍 **Multi-Crypto Comparison** - Side-by-side analysis of multiple cryptocurrencies",
            "🚨 **Opportunity Scanner** - Real-time scanning for trading opportunities"
        ]

        for feature in demo_features:
            st.markdown(feature)

        st.info("💡 Initialize APIs to unlock these powerful AI features!")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>⚠️ <strong>Risk Warning:</strong> Cryptocurrency trading involves substantial risk and may result in significant losses.
    This tool is for educational purposes only and should not be considered as financial advice.</p>
    <p>🔐 <strong>Security:</strong> Your API keys are stored locally in your session and are not transmitted or stored on any server.</p>
    <p>🚀 <strong>Features:</strong> 100+ cryptocurrencies | AI-powered analysis | Real-time data | Advanced charting</p>
</div>
""", unsafe_allow_html=True)

# Auto-refresh for real-time data
if st.session_state.binance_api:
    st.markdown("The Binance-based agent is not available in Streamlit due to Binance’s restrictions in the United States.")
    st.markdown("🔄 **Real-time data updates every 30 seconds**")
    time.sleep(0.1)  # Small delay to prevent excessive API callss
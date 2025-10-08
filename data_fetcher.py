"""
Stock Data Fetcher Module
Handles fetching stock data from Yahoo Finance with error handling and data validation
"""

import yfinance as yf
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta
import numpy as np

class StockDataFetcher:
    """
    A class to fetch and process stock data from Yahoo Finance
    """

    def __init__(self):
        self.ticker = None
        self.data = {}

    def get_stock_data(self, symbol, period="1y"):
        """
        Fetch comprehensive stock data for a given symbol

        Args:
            symbol (str): Stock ticker symbol
            period (str): Time period for historical data

        Returns:
            dict: Comprehensive stock data or None if failed
        """
        try:
            # Create ticker object
            self.ticker = yf.Ticker(symbol)

            # Fetch basic info
            info = self.ticker.info

            # Validate that we got valid data
            if not info or 'symbol' not in info:
                st.error(f"Invalid ticker symbol: {symbol}")
                return None

            # Fetch historical data
            history = self.ticker.history(period=period)

            if history.empty:
                st.error(f"No historical data found for {symbol}")
                return None

            # Fetch financial data
            financials = self.get_financial_data()

            # Compile all data
            stock_data = {
                'symbol': symbol,
                'info': info,
                'history': history,
                'financials': financials,
                'last_updated': datetime.now()
            }

            return stock_data

        except Exception as e:
            st.error(f"Error fetching data for {symbol}: {str(e)}")
            return None

    def get_financial_data(self):
        """
        Fetch financial statements and key metrics

        Returns:
            dict: Financial data or empty dict if failed
        """
        financial_data = {}

        try:
            # Income Statement
            try:
                income_stmt = self.ticker.financials
                financial_data['income_statement'] = income_stmt
            except:
                financial_data['income_statement'] = pd.DataFrame()

            # Balance Sheet
            try:
                balance_sheet = self.ticker.balance_sheet
                financial_data['balance_sheet'] = balance_sheet
            except:
                financial_data['balance_sheet'] = pd.DataFrame()

            # Cash Flow Statement
            try:
                cash_flow = self.ticker.cashflow
                financial_data['cash_flow'] = cash_flow
            except:
                financial_data['cash_flow'] = pd.DataFrame()

            # Quarterly data
            try:
                quarterly_financials = self.ticker.quarterly_financials
                financial_data['quarterly_financials'] = quarterly_financials
            except:
                financial_data['quarterly_financials'] = pd.DataFrame()

        except Exception as e:
            st.warning(f"Some financial data could not be retrieved: {str(e)}")

        return financial_data

    def get_key_metrics(self, info):
        """
        Extract key financial metrics from stock info

        Args:
            info (dict): Stock info from yfinance

        Returns:
            dict: Key financial metrics
        """
        metrics = {}

        # Price metrics
        metrics['current_price'] = info.get('currentPrice', 0)
        metrics['previous_close'] = info.get('previousClose', 0)
        metrics['day_high'] = info.get('dayHigh', 0)
        metrics['day_low'] = info.get('dayLow', 0)

        # Valuation metrics
        metrics['market_cap'] = info.get('marketCap', 0)
        metrics['pe_ratio'] = info.get('trailingPE', None)
        metrics['forward_pe'] = info.get('forwardPE', None)
        metrics['pb_ratio'] = info.get('priceToBook', None)
        metrics['ps_ratio'] = info.get('priceToSalesTrailing12Months', None)
        metrics['peg_ratio'] = info.get('pegRatio', None)

        # Financial health metrics
        metrics['debt_to_equity'] = info.get('debtToEquity', None)
        metrics['return_on_equity'] = info.get('returnOnEquity', None)
        metrics['return_on_assets'] = info.get('returnOnAssets', None)

        # Growth and profitability
        metrics['revenue_growth'] = info.get('revenueGrowth', None)
        metrics['earnings_growth'] = info.get('earningsGrowth', None)
        metrics['profit_margins'] = info.get('profitMargins', None)
        metrics['operating_margins'] = info.get('operatingMargins', None)

        # Dividend information
        metrics['dividend_yield'] = info.get('dividendYield', None)
        metrics['dividend_rate'] = info.get('dividendRate', None)
        metrics['payout_ratio'] = info.get('payoutRatio', None)

        # Cash flow metrics
        metrics['operating_cash_flow'] = info.get('operatingCashflow', None)
        metrics['free_cash_flow'] = info.get('freeCashflow', None)

        # Shares information
        metrics['shares_outstanding'] = info.get('sharesOutstanding', None)
        metrics['float_shares'] = info.get('floatShares', None)

        # Risk metrics
        metrics['beta'] = info.get('beta', None)

        # 52-week range
        metrics['fifty_two_week_high'] = info.get('fiftyTwoWeekHigh', None)
        metrics['fifty_two_week_low'] = info.get('fiftyTwoWeekLow', None)

        return metrics

    def get_sector_industry_data(self, info):
        """
        Extract sector and industry information

        Args:
            info (dict): Stock info from yfinance

        Returns:
            dict: Sector and industry data
        """
        return {
            'sector': info.get('sector', 'N/A'),
            'industry': info.get('industry', 'N/A'),
            'business_summary': info.get('longBusinessSummary', 'N/A'),
            'country': info.get('country', 'N/A'),
            'website': info.get('website', 'N/A'),
            'full_time_employees': info.get('fullTimeEmployees', None)
        }

    def validate_data_quality(self, stock_data):
        """
        Validate the quality and completeness of fetched data

        Args:
            stock_data (dict): Stock data to validate

        Returns:
            dict: Validation results with warnings
        """
        warnings = []
        quality_score = 100

        info = stock_data.get('info', {})

        # Check for essential price data
        if not info.get('currentPrice'):
            warnings.append("Current price not available")
            quality_score -= 20

        # Check for valuation metrics
        if not info.get('trailingPE'):
            warnings.append("P/E ratio not available")
            quality_score -= 10

        if not info.get('priceToBook'):
            warnings.append("P/B ratio not available")
            quality_score -= 10

        # Check for financial data
        financials = stock_data.get('financials', {})
        if not financials or all(df.empty for df in financials.values()):
            warnings.append("Limited financial statement data")
            quality_score -= 15

        # Check historical data
        history = stock_data.get('history', pd.DataFrame())
        if history.empty:
            warnings.append("No historical price data")
            quality_score -= 25
        elif len(history) < 30:
            warnings.append("Limited historical data (less than 30 days)")
            quality_score -= 10

        return {
            'quality_score': max(0, quality_score),
            'warnings': warnings,
            'data_completeness': {
                'has_price': bool(info.get('currentPrice')),
                'has_financials': bool(financials and not all(df.empty for df in financials.values())),
                'has_history': not history.empty,
                'has_valuation_metrics': bool(info.get('trailingPE') or info.get('priceToBook'))
            }
        }

    def get_comparable_companies(self, symbol, sector):
        """
        Get a list of comparable companies in the same sector
        (This is a placeholder - in real implementation, you might use a sector mapping)

        Args:
            symbol (str): Current stock symbol
            sector (str): Sector of the company

        Returns:
            list: List of comparable company symbols
        """
        # Sector to companies mapping (simplified example)
        sector_companies = {
            'Technology': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA'],
            'Healthcare': ['JNJ', 'PFE', 'UNH', 'ABBV', 'MRK', 'TMO'],
            'Financial Services': ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'C'],
            'Consumer Cyclical': ['AMZN', 'TSLA', 'HD', 'NKE', 'MCD', 'SBUX'],
            'Communication Services': ['GOOGL', 'META', 'DIS', 'NFLX', 'T', 'VZ'],
            'Industrials': ['BA', 'CAT', 'GE', 'MMM', 'LMT', 'RTX'],
            'Consumer Defensive': ['PG', 'KO', 'PEP', 'WMT', 'COST', 'CL'],
            'Energy': ['XOM', 'CVX', 'COP', 'EOG', 'SLB', 'MPC'],
            'Utilities': ['NEE', 'D', 'SO', 'DUK', 'AEP', 'EXC'],
            'Real Estate': ['AMT', 'PLD', 'CCI', 'EQIX', 'SPG', 'O'],
            'Materials': ['LIN', 'APD', 'SHW', 'FCX', 'NEM', 'DOW']
        }

        companies = sector_companies.get(sector, [])
        # Remove current symbol from comparables
        return [comp for comp in companies if comp != symbol][:5]  # Return top 5

    @staticmethod
    def format_large_number(number):
        """
        Format large numbers for display (e.g., 1.5B, 250M)

        Args:
            number: Number to format

        Returns:
            str: Formatted number string
        """
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

    @staticmethod
    def calculate_technical_indicators(history):
        """
        Calculate basic technical indicators

        Args:
            history: Historical price data DataFrame

        Returns:
            dict: Technical indicators
        """
        if history.empty or len(history) < 20:
            return {}

        indicators = {}

        # Moving averages
        indicators['sma_20'] = history['Close'].rolling(window=20).mean().iloc[-1]
        indicators['sma_50'] = history['Close'].rolling(window=50).mean().iloc[-1] if len(history) >= 50 else None

        # Volatility (annualized)
        returns = history['Close'].pct_change().dropna()
        indicators['volatility'] = returns.std() * np.sqrt(252)

        # RSI (simplified version)
        delta = history['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        indicators['rsi'] = 100 - (100 / (1 + rs)).iloc[-1] if len(history) >= 14 else None

        return indicators
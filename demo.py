"""
Demo Script - Stock Valuation Tool
Shows how to use the modules programmatically without the UI
"""

from data_fetcher import StockDataFetcher
from valuation_methods import ValuationCalculator
import pandas as pd

def demo_stock_analysis(symbol="AAPL"):
    """
    Demonstrate stock analysis for a given symbol

    Args:
        symbol (str): Stock ticker symbol
    """
    print(f"\n{'='*50}")
    print(f"📊 STOCK ANALYSIS FOR {symbol}")
    print(f"{'='*50}")

    # Initialize data fetcher
    print("🔄 Fetching stock data...")
    fetcher = StockDataFetcher()

    # Get stock data
    stock_data = fetcher.get_stock_data(symbol, period="1y")

    if not stock_data:
        print(f"❌ Failed to fetch data for {symbol}")
        return

    # Display basic info
    info = stock_data['info']
    print(f"\n📋 COMPANY INFORMATION:")
    print(f"Name: {info.get('longName', 'N/A')}")
    print(f"Sector: {info.get('sector', 'N/A')}")
    print(f"Industry: {info.get('industry', 'N/A')}")
    print(f"Current Price: ${info.get('currentPrice', 0):.2f}")
    print(f"Market Cap: ${info.get('marketCap', 0):,.0f}")

    # Initialize valuation calculator
    calculator = ValuationCalculator(stock_data)

    # Calculate all valuations
    print(f"\n🧮 VALUATION ANALYSIS:")
    results = calculator.calculate_all_valuations()

    # Display results in a nice format
    for method, result in results.items():
        if isinstance(result, dict):
            value = result.get('value', 'N/A')
            status = result.get('status', 'Unknown')

            # Color coding for status
            status_emoji = {
                'Undervalued': '🟢',
                'Fair': '🟡', 
                'Overvalued': '🔴',
                'Insufficient Data': '⚪',
                'No Dividend': '⚫'
            }.get(status, '❓')

            print(f"{method:.<25} {value:>10} {status_emoji} {status}")

    # Financial health analysis
    health = calculator.analyze_financial_health()
    print(f"\n💊 FINANCIAL HEALTH:")
    print(f"Health Score: {health['health_score']}/100")
    print(f"Recommendation: {health['recommendation']}")

    if health['strengths']:
        print("\n✅ Strengths:")
        for strength in health['strengths']:
            print(f"  • {strength}")

    if health['warnings']:
        print("\n⚠️  Warnings:")
        for warning in health['warnings']:
            print(f"  • {warning}")

    # DCF Analysis with different scenarios
    print(f"\n💰 DCF ANALYSIS (Multiple Scenarios):")
    scenarios = [
        (5, 10, "Conservative"),
        (8, 10, "Moderate"), 
        (12, 10, "Optimistic")
    ]

    for growth, discount, scenario_name in scenarios:
        dcf_result = calculator.calculate_dcf_value(growth, discount)
        if dcf_result and dcf_result['value'] != 'N/A':
            print(f"{scenario_name:.<15} {dcf_result['value']:>10} ({dcf_result['status']})")

    # Valuation summary
    summary = calculator.get_valuation_summary(results)
    print(f"\n📊 OVERALL ASSESSMENT:")
    print(f"Status: {summary['overall_status']}")
    print(f"Confidence: {summary['confidence']}")
    print(f"Recommendation: {summary['recommendation']}")

    # Historical performance
    history = stock_data['history']
    if not history.empty:
        print(f"\n📈 RECENT PERFORMANCE:")
        recent_return = ((history['Close'].iloc[-1] - history['Close'].iloc[0]) / history['Close'].iloc[0]) * 100
        print(f"1-Year Return: {recent_return:.1f}%")

        volatility = history['Close'].pct_change().std() * (252**0.5) * 100
        print(f"Annualized Volatility: {volatility:.1f}%")

    print(f"\n{'='*50}")
    print(f"✅ Analysis completed for {symbol}")
    print(f"{'='*50}")

def compare_stocks(symbols=["AAPL", "MSFT", "GOOGL"]):
    """
    Compare multiple stocks side by side

    Args:
        symbols (list): List of stock ticker symbols
    """
    print(f"\n{'='*60}")
    print(f"🔄 COMPARING MULTIPLE STOCKS")
    print(f"{'='*60}")

    comparison_data = []
    fetcher = StockDataFetcher()

    for symbol in symbols:
        print(f"\nProcessing {symbol}...")
        stock_data = fetcher.get_stock_data(symbol, period="1y")

        if stock_data:
            info = stock_data['info']
            calculator = ValuationCalculator(stock_data)
            results = calculator.calculate_all_valuations()

            # Extract key metrics for comparison
            row = {
                'Symbol': symbol,
                'Name': info.get('longName', 'N/A')[:20] + '...' if len(info.get('longName', '')) > 20 else info.get('longName', 'N/A'),
                'Price': f"${info.get('currentPrice', 0):.2f}",
                'P/E': results.get('P/E Ratio', {}).get('raw_value', 'N/A'),
                'P/B': results.get('P/B Ratio', {}).get('raw_value', 'N/A'),
                'P/S': results.get('P/S Ratio', {}).get('raw_value', 'N/A'),
                'Div Yield': info.get('dividendYield', 0) if info.get('dividendYield') else 0,
                'Market Cap (B)': info.get('marketCap', 0) / 1e9 if info.get('marketCap') else 0
            }
            comparison_data.append(row)

    # Create and display comparison table
    if comparison_data:
        df = pd.DataFrame(comparison_data)
        print(f"\n📊 COMPARISON TABLE:")
        print(df.to_string(index=False, float_format='%.2f'))

    print(f"\n{'='*60}")
    print("✅ Comparison completed")
    print(f"{'='*60}")

if __name__ == "__main__":
    # Demo single stock analysis
    demo_stock_analysis("AAPL")

    # Demo stock comparison
    compare_stocks(["AAPL", "MSFT", "GOOGL"])

    print("\n🎉 Demo completed!")
    print("📝 To use the full UI application, run: streamlit run main.py")
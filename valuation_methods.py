"""
Enhanced Valuation Methods Module - Version 2.0
Added Peter Lynch, Warren Buffett, Benjamin Graham, and Hedge Fund approaches
"""

import pandas as pd
import numpy as np

from market_data import MarketAveragesFetcher
from datetime import datetime
import streamlit as st

class EnhancedValuationCalculator:
    """
    Enhanced class with legendary investor methods and hedge fund techniques
    """

    def __init__(self, stock_data):
        self.stock_data = stock_data
        self.info = stock_data.get('info', {})
        self.history = stock_data.get('history', pd.DataFrame())
        self.financials = stock_data.get('financials', {})

    def calculate_all_valuations(self):
        """Calculate ALL valuation methods including legendary approaches"""
        results = {}

        # Original basic methods
        results['P/E Ratio'] = self.calculate_pe_ratio()
        results['P/B Ratio'] = self.calculate_pb_ratio()
        results['P/S Ratio'] = self.calculate_ps_ratio()
        results['PEG Ratio'] = self.calculate_peg_ratio()
        results['Dividend Yield'] = self.calculate_dividend_yield()
        results['P/CF Ratio'] = self.calculate_price_to_cash_flow()
        results['EV/EBITDA'] = self.calculate_ev_ebitda()
        results['Asset-Based Value'] = self.calculate_asset_based_value()

        # LEGENDARY INVESTOR METHODS
        results['Peter Lynch GARP'] = self.calculate_peter_lynch_garp()
        results['Lynch Fair Value'] = self.calculate_lynch_fair_value()
        results['Warren Buffett Score'] = self.calculate_buffett_criteria()
        results['Buffett Intrinsic Value'] = self.calculate_buffett_intrinsic_value()
        results['Benjamin Graham Net-Net'] = self.calculate_graham_net_net()
        results['Graham Defensive'] = self.calculate_graham_defensive_criteria()
        results['Magic Formula (Greenblatt)'] = self.calculate_magic_formula()

        # HEDGE FUND METHODS
        results['Momentum Score'] = self.calculate_momentum_factor()
        results['Quality Score'] = self.calculate_quality_factor()
        results['Mean Reversion'] = self.calculate_mean_reversion()
        results['Multi-Factor Model'] = self.calculate_multi_factor_score()

        return results

    # ================== PETER LYNCH METHODS ==================

    def calculate_pe_ratio(self):
        """Calculate Price-to-Earnings ratio analysis"""
        try:
            current_price = self.info.get('currentPrice', 0)
            pe_ratio = self.info.get('trailingPE', None)

            if not pe_ratio or pe_ratio <= 0:
                return {
                    'value': 'N/A',
                    'status': 'Insufficient Data',
                    'description': 'P/E ratio not available or company has negative earnings'
                }

            industry_name = self.info.get('industry', '')
            sector_name = self.info.get('sector', '')
            industry_metrics = MarketAveragesFetcher.get_industry_metrics(industry_name)
            sector_metrics = MarketAveragesFetcher.get_sector_metrics(sector_name)

            benchmark_sources = []
            comparisons = []

            if industry_metrics.get('pe'):
                comparisons.append(
                    f"{industry_metrics.get('name', 'Industry')} Avg: {industry_metrics['pe']:.2f}x"
                )
                if industry_metrics.get('source'):
                    benchmark_sources.append(industry_metrics['source'])

            if sector_metrics.get('pe'):
                comparisons.append(
                    f"{sector_metrics.get('name', 'Sector')} Avg: {sector_metrics['pe']:.2f}x"
                )
                if sector_metrics.get('source'):
                    benchmark_sources.append(sector_metrics['source'])

            benchmark_pe = (
                industry_metrics.get('pe')
                or sector_metrics.get('pe')
                or self.get_industry_average_pe(sector_name)
            )

            if benchmark_pe:
                if pe_ratio < benchmark_pe * 0.8:
                    status = 'Undervalued'
                elif pe_ratio > benchmark_pe * 1.3:
                    status = 'Overvalued'
                else:
                    status = 'Fair'
            else:
                status = 'Fair'

            description_parts = [f'P/E: {pe_ratio:.2f}']
            if comparisons:
                description_parts.append(' vs '.join(comparisons))
            if benchmark_sources:
                unique_sources = list(dict.fromkeys(benchmark_sources))
                description_parts.append(f"Source: {', '.join(unique_sources)}")

            return {
                'value': f"{pe_ratio:.2f}x",
                'status': status,
                'description': ' | '.join(description_parts),
                'raw_value': pe_ratio,
                'industry_avg': industry_metrics.get('pe'),
                'industry_source': industry_metrics.get('source'),
                'sector_avg_pe': sector_metrics.get('pe'),
                'sector_source': sector_metrics.get('source'),
            }

        except Exception as e:
            return {
                'value': 'Error',
                'status': 'Error',
                'description': f'Error calculating P/E: {str(e)}'
            }

    def calculate_pb_ratio(self):
        """Calculate Price-to-Book ratio analysis"""
        try:
            pb_ratio = self.info.get('priceToBook', None)

            if not pb_ratio or pb_ratio <= 0:
                return {
                    'value': 'N/A',
                    'status': 'Insufficient Data',
                    'description': 'P/B ratio not available'
                }

            # General P/B evaluation (simplified)
            if pb_ratio < 1:
                status = 'Undervalued'
            elif pb_ratio > 3:
                status = 'Overvalued'
            else:
                status = 'Fair'

            return {
                'value': f"{pb_ratio:.2f}x",
                'status': status,
                'description': f'P/B ratio of {pb_ratio:.2f}. Value <1 often indicates undervaluation.',
                'raw_value': pb_ratio
            }

        except Exception as e:
            return {
                'value': 'Error',
                'status': 'Error',
                'description': f'Error calculating P/B: {str(e)}'
            }

    def calculate_ps_ratio(self):
        """Calculate Price-to-Sales ratio analysis"""
        try:
            ps_ratio = self.info.get('priceToSalesTrailing12Months', None)

            if not ps_ratio or ps_ratio <= 0:
                # Try to calculate manually
                market_cap = self.info.get('marketCap', None)
                revenue = self.info.get('totalRevenue', None)

                if market_cap and revenue and revenue > 0:
                    ps_ratio = market_cap / revenue
                else:
                    return {
                        'value': 'N/A',
                        'status': 'Insufficient Data',
                        'description': 'P/S ratio not available'
                    }

            sector_name = self.info.get('sector', '')
            industry_name = self.info.get('industry', '')

            sector_metrics = MarketAveragesFetcher.get_sector_metrics(sector_name)
            industry_metrics = MarketAveragesFetcher.get_industry_metrics(industry_name)

            benchmark_ps = (
                sector_metrics.get('ps')
                or industry_metrics.get('ps')
                or self.get_sector_average_ps(sector_name)
            )

            if benchmark_ps:
                if ps_ratio < benchmark_ps * 0.7:
                    status = 'Undervalued'
                elif ps_ratio > benchmark_ps * 1.5:
                    status = 'Overvalued'
                else:
                    status = 'Fair'
            else:
                status = 'Fair'

            comparison_parts = []
            if sector_metrics.get('ps') is not None:
                comparison_parts.append(
                    f"{sector_metrics.get('name', 'Sector')} Avg: {sector_metrics['ps']:.2f}x"
                )
            if industry_metrics.get('ps') is not None:
                comparison_parts.append(
                    f"{industry_metrics.get('name', 'Industry')} Avg: {industry_metrics['ps']:.2f}x"
                )

            sources = [
                source
                for source in (
                    sector_metrics.get('source'),
                    industry_metrics.get('source'),
                )
                if source
            ]

            description = f'P/S: {ps_ratio:.2f}'
            if comparison_parts:
                description += ' vs ' + ' & '.join(comparison_parts)
            if sources:
                unique_sources = list(dict.fromkeys(sources))
                description += f" | Source: {', '.join(unique_sources)}"

            return {
                'value': f"{ps_ratio:.2f}x",
                'status': status,
                'description': description,
                'raw_value': ps_ratio,
                'sector_avg': sector_metrics.get('ps'),
                'sector_source': sector_metrics.get('source'),
                'industry_ps': industry_metrics.get('ps'),
                'industry_source': industry_metrics.get('source'),
            }

        except Exception as e:
            return {
                'value': 'Error',
                'status': 'Error',
                'description': f'Error calculating P/S: {str(e)}'
            }

    def calculate_peg_ratio(self):
        """Calculate PEG (P/E to Growth) ratio analysis"""
        try:
            pe_ratio = self.info.get('trailingPE', None)
            peg_ratio = self.info.get('pegRatio', None)

            if not peg_ratio and pe_ratio:
                # Try to calculate using earnings growth
                earnings_growth = self.info.get('earningsGrowth', None)
                if earnings_growth and earnings_growth > 0:
                    peg_ratio = pe_ratio / (earnings_growth * 100)

            if not peg_ratio or peg_ratio <= 0:
                return {
                    'value': 'N/A',
                    'status': 'Insufficient Data',
                    'description': 'PEG ratio not available - requires earnings growth data'
                }

            # PEG ratio evaluation
            if peg_ratio < 0.5:
                status = 'Undervalued'
            elif peg_ratio < 1.0:
                status = 'Fair'
            elif peg_ratio < 1.5:
                status = 'Overvalued'
            else:
                status = 'Overvalued'

            return {
                'value': f"{peg_ratio:.2f}",
                'status': status,
                'description': f'PEG of {peg_ratio:.2f}. Values <1.0 generally indicate good value.',
                'raw_value': peg_ratio
            }

        except Exception as e:
            return {
                'value': 'Error',
                'status': 'Error',
                'description': f'Error calculating PEG: {str(e)}'
            }

    def calculate_dividend_yield(self):
        """Calculate dividend yield analysis"""
        try:
            dividend_yield = self.info.get('dividendYield', None)

            if not dividend_yield:
                return {
                    'value': 'N/A',
                    'status': 'No Dividend',
                    'description': 'Company does not pay dividends'
                }

            dividend_yield_pct = dividend_yield

            # Dividend yield evaluation
            sector = self.info.get('sector', '')

            if dividend_yield_pct > 4:
                status = 'High Yield'
            elif dividend_yield_pct > 2:
                status = 'Moderate Yield'
            else:
                status = 'Low Yield'

            return {
                'value': f"{dividend_yield_pct:.2f}%",
                'status': status,
                'description': f'Dividend yield of {dividend_yield_pct:.2f}%',
                'raw_value': dividend_yield_pct
            }

        except Exception as e:
            return {
                'value': 'Error',
                'status': 'Error',
                'description': f'Error calculating dividend yield: {str(e)}'
            }

    def calculate_price_to_cash_flow(self):
        """Calculate Price-to-Cash Flow ratio"""
        try:
            market_cap = self.info.get('marketCap', None)
            operating_cash_flow = self.info.get('operatingCashflow', None)

            if not market_cap or not operating_cash_flow or operating_cash_flow <= 0:
                return {
                    'value': 'N/A',
                    'status': 'Insufficient Data',
                    'description': 'Cash flow data not available'
                }

            pcf_ratio = market_cap / operating_cash_flow

            # P/CF evaluation (general guidelines)
            if pcf_ratio < 10:
                status = 'Undervalued'
            elif pcf_ratio < 20:
                status = 'Fair'
            else:
                status = 'Overvalued'

            return {
                'value': f"{pcf_ratio:.2f}x",
                'status': status,
                'description': f'P/CF ratio of {pcf_ratio:.2f}. Lower values generally better.',
                'raw_value': pcf_ratio
            }

        except Exception as e:
            return {
                'value': 'Error',
                'status': 'Error',
                'description': f'Error calculating P/CF: {str(e)}'
            }

    def calculate_ev_ebitda(self):
        """Calculate Enterprise Value to EBITDA ratio"""
        try:
            market_cap = self.info.get('marketCap', None)
            total_debt = self.info.get('totalDebt', 0)
            total_cash = self.info.get('totalCash', 0)
            ebitda = self.info.get('ebitda', None)

            if not market_cap or not ebitda or ebitda <= 0:
                return {
                    'value': 'N/A',
                    'status': 'Insufficient Data',
                    'description': 'EBITDA or market data not available'
                }

            enterprise_value = market_cap + total_debt - total_cash
            ev_ebitda = enterprise_value / ebitda

            # EV/EBITDA evaluation
            if ev_ebitda < 10:
                status = 'Undervalued'
            elif ev_ebitda < 15:
                status = 'Fair'
            else:
                status = 'Overvalued'

            return {
                'value': f"{ev_ebitda:.2f}x",
                'status': status,
                'description': f'EV/EBITDA of {ev_ebitda:.2f}. Lower values generally indicate better value.',
                'raw_value': ev_ebitda,
                'enterprise_value': enterprise_value
            }

        except Exception as e:
            return {
                'value': 'Error',
                'status': 'Error',
                'description': f'Error calculating EV/EBITDA: {str(e)}'
            }

    def calculate_asset_based_value(self):
        """Calculate asset-based valuation (Book Value approach)"""
        try:
            total_assets = self.info.get('totalAssets', None)
            total_liabilities = self.info.get('totalLiab', None)
            shares_outstanding = self.info.get('sharesOutstanding', None)

            if not all([total_assets, total_liabilities, shares_outstanding]):
                return {
                    'value': 'N/A',
                    'status': 'Insufficient Data',
                    'description': 'Balance sheet data not available'
                }

            book_value = total_assets - total_liabilities
            book_value_per_share = book_value / shares_outstanding
            current_price = self.info.get('currentPrice', 0)

            if current_price > 0:
                price_to_book = current_price / book_value_per_share

                if price_to_book < 0.8:
                    status = 'Undervalued'
                elif price_to_book < 1.5:
                    status = 'Fair'
                else:
                    status = 'Overvalued'
            else:
                status = 'Unknown'

            return {
                'value': f"${book_value_per_share:.2f}",
                'status': status,
                'description': f'Book value per share: ${book_value_per_share:.2f}',
                'raw_value': book_value_per_share,
                'total_book_value': book_value
            }

        except Exception as e:
            return {
                'value': 'Error',
                'status': 'Error',
                'description': f'Error calculating asset-based value: {str(e)}'
            }


    def calculate_dcf_value(self, growth_rate, discount_rate, terminal_growth=2.5):
        """
        Simplified DCF calculation

        Args:
            growth_rate (float): Expected growth rate (%)
            discount_rate (float): Discount rate/WACC (%)
            terminal_growth (float): Terminal growth rate (%)

        Returns:
            dict: DCF valuation results
        """
        try:
            free_cash_flow = self.info.get('freeCashflow', None)
            if not free_cash_flow:
                # Use operating cash flow as proxy
                free_cash_flow = self.info.get('operatingCashflow', None)

            shares_outstanding = self.info.get('sharesOutstanding', None)

            if not free_cash_flow or not shares_outstanding or free_cash_flow <= 0:
                return {
                    'value': 'N/A',
                    'status': 'Insufficient Data',
                    'description': 'Cash flow data not available for DCF calculation'
                }

            # Simple DCF calculation (5-year projection + terminal value)
            growth_rate_decimal = growth_rate / 100
            discount_rate_decimal = discount_rate / 100
            terminal_growth_decimal = terminal_growth / 100

            # Project cash flows for 5 years
            projected_cf = []
            for year in range(1, 6):
                cf = free_cash_flow * ((1 + growth_rate_decimal) ** year)
                pv_cf = cf / ((1 + discount_rate_decimal) ** year)
                projected_cf.append(pv_cf)

            # Terminal value
            terminal_cf = free_cash_flow * ((1 + growth_rate_decimal) ** 6)
            terminal_value = terminal_cf / (discount_rate_decimal - terminal_growth_decimal)
            pv_terminal_value = terminal_value / ((1 + discount_rate_decimal) ** 5)

            # Total enterprise value
            enterprise_value = sum(projected_cf) + pv_terminal_value

            # Equity value (simplified - not accounting for net debt)
            equity_value = enterprise_value
            value_per_share = equity_value / shares_outstanding

            current_price = self.info.get('currentPrice', 0)
            if current_price > 0:
                upside = (value_per_share - current_price) / current_price * 100
                if upside > 20:
                    status = 'Undervalued'
                elif upside > -10:
                    status = 'Fair'
                else:
                    status = 'Overvalued'
            else:
                status = 'Unknown'

            return {
                'value': f"${value_per_share:.2f}",
                'status': status,
                'description': f'DCF value: ${value_per_share:.2f} (Growth: {growth_rate}%, Discount: {discount_rate}%)',
                'raw_value': value_per_share,
                'enterprise_value': enterprise_value,
                'upside_percentage': upside if current_price > 0 else None
            }

        except Exception as e:
            return {
                'value': 'Error',
                'status': 'Error',
                'description': f'Error in DCF calculation: {str(e)}'
            }

    def get_industry_average_pe(self, sector):
        """Fallback industry average P/E ratios when live data is unavailable."""

        if sector:
            fallback = MarketAveragesFetcher.SECTOR_FALLBACK.get(sector.strip().lower())
            if fallback and fallback.get('pe') is not None:
                return fallback['pe']

        return 18.0  # Conservative default

    def get_sector_average_ps(self, sector):
        """Fallback sector average P/S ratios when live data is unavailable."""

        if sector:
            fallback = MarketAveragesFetcher.SECTOR_FALLBACK.get(sector.strip().lower())
            if fallback and fallback.get('ps') is not None:
                return fallback['ps']

        return 2.5  # Conservative default

    def analyze_financial_health(self):
        """Analyze overall financial health of the company"""
        try:
            health_score = 100
            warnings = []
            strengths = []

            # Debt analysis
            debt_to_equity = self.info.get('debtToEquity', None)
            if debt_to_equity:
                if debt_to_equity > 100:  # High debt
                    health_score -= 15
                    warnings.append(f"High debt-to-equity ratio: {debt_to_equity:.1f}%")
                elif debt_to_equity < 30:  # Conservative debt
                    strengths.append(f"Conservative debt level: {debt_to_equity:.1f}%")

            # Profitability analysis
            profit_margins = self.info.get('profitMargins', None)
            if profit_margins:
                if profit_margins > 0.20:  # High margins
                    strengths.append(f"Strong profit margins: {profit_margins*100:.1f}%")
                elif profit_margins < 0.05:  # Low margins
                    health_score -= 10
                    warnings.append(f"Low profit margins: {profit_margins*100:.1f}%")

            # ROE analysis
            roe = self.info.get('returnOnEquity', None)
            if roe:
                if roe > 0.15:  # Strong ROE
                    strengths.append(f"Strong ROE: {roe*100:.1f}%")
                elif roe < 0.05:  # Weak ROE
                    health_score -= 10
                    warnings.append(f"Weak ROE: {roe*100:.1f}%")

            # Cash position
            total_cash = self.info.get('totalCash', 0)
            total_debt = self.info.get('totalDebt', 0)
            if total_cash > total_debt:
                strengths.append("Strong cash position (cash > debt)")

            return {
                'health_score': max(0, health_score),
                'warnings': warnings,
                'strengths': strengths,
                'recommendation': self.get_health_recommendation(health_score)
            }

        except Exception as e:
            return {
                'health_score': 0,
                'warnings': [f"Error analyzing financial health: {str(e)}"],
                'strengths': [],
                'recommendation': 'Unable to assess'
            }

    def get_health_recommendation(self, score):
        """Get recommendation based on health score"""
        if score >= 80:
            return "Strong - Good financial health"
        elif score >= 60:
            return "Good - Generally healthy with minor concerns"
        elif score >= 40:
            return "Moderate - Some financial concerns to monitor"
        else:
            return "Weak - Significant financial concerns"

    def get_valuation_summary(self, results):
        """
        Generate an overall valuation summary

        Args:
            results (dict): Results from all valuation methods

        Returns:
            dict: Summary of valuations
        """
        try:
            # Count status occurrences
            status_counts = {'Undervalued': 0, 'Fair': 0, 'Overvalued': 0, 'Unknown': 0}

            valid_methods = 0
            for method, result in results.items():
                if isinstance(result, dict) and result.get('status') in status_counts:
                    status_counts[result['status']] += 1
                    valid_methods += 1

            if valid_methods == 0:
                return {
                    'overall_status': 'Insufficient Data',
                    'confidence': 'Low',
                    'recommendation': 'Unable to provide valuation assessment'
                }

            # Determine overall status
            undervalued_pct = status_counts['Undervalued'] / valid_methods
            overvalued_pct = status_counts['Overvalued'] / valid_methods

            if undervalued_pct >= 0.6:
                overall_status = 'Undervalued'
                confidence = 'High' if undervalued_pct >= 0.8 else 'Medium'
            elif overvalued_pct >= 0.6:
                overall_status = 'Overvalued'
                confidence = 'High' if overvalued_pct >= 0.8 else 'Medium'
            else:
                overall_status = 'Fair'
                confidence = 'Medium'

            # Generate recommendation
            if overall_status == 'Undervalued':
                recommendation = 'Consider buying - multiple methods suggest undervaluation'
            elif overall_status == 'Overvalued':
                recommendation = 'Consider selling or avoid - multiple methods suggest overvaluation'
            else:
                recommendation = 'Hold or neutral - mixed signals from valuation methods'

            return {
                'overall_status': overall_status,
                'confidence': confidence,
                'recommendation': recommendation,
                'method_breakdown': status_counts,
                'valid_methods_count': valid_methods
            }

        except Exception as e:
            return {
                'overall_status': 'Error',
                'confidence': 'None',
                'recommendation': f'Error generating summary: {str(e)}'
            }
            
    def calculate_peter_lynch_garp(self):
        """Peter Lynch's GARP (Growth At Reasonable Price) approach"""
        try:
            pe_ratio = self.info.get('trailingPE', None)
            growth_rate = self.info.get('earningsGrowth', None)

            if not pe_ratio or not growth_rate:
                return {
                    'value': 'N/A',
                    'status': 'Insufficient Data',
                    'description': 'PE ratio or earnings growth not available for GARP calculation'
                }

            # Convert growth rate to percentage
            growth_pct = growth_rate * 100 if growth_rate < 1 else growth_rate

            # PEG ratio (core of GARP)
            peg_ratio = pe_ratio / growth_pct if growth_pct > 0 else None

            if not peg_ratio:
                return {
                    'value': 'N/A',
                    'status': 'Invalid Data',
                    'description': 'Negative or zero growth rate'
                }

            # Peter Lynch GARP criteria
            if peg_ratio < 0.5:
                status = 'Exceptional Buy'
            elif peg_ratio < 1.0:
                status = 'Good Buy'
            elif peg_ratio < 1.5:
                status = 'Fair'
            elif peg_ratio < 2.0:
                status = 'Overvalued'
            else:
                status = 'Avoid'

            # Additional Lynch criteria
            debt_to_equity = self.info.get('debtToEquity', 0)
            roe = self.info.get('returnOnEquity', 0)

            bonus_criteria = []
            if debt_to_equity < 30:
                bonus_criteria.append("Low Debt ✓")
            if roe and roe > 0.15:
                bonus_criteria.append("High ROE ✓")

            description = f"PEG: {peg_ratio:.2f} (PE: {pe_ratio:.1f}, Growth: {growth_pct:.1f}%)"
            if bonus_criteria:
                description += f" | {', '.join(bonus_criteria)}"

            return {
                'value': f"{peg_ratio:.2f}",
                'status': status,
                'description': description,
                'raw_value': peg_ratio,
                'pe_ratio': pe_ratio,
                'growth_rate': growth_pct
            }

        except Exception as e:
            return {
                'value': 'Error',
                'status': 'Error',
                'description': f'Error in GARP calculation: {str(e)}'
            }

    def calculate_lynch_fair_value(self):
        """Peter Lynch Fair Value Formula: (Growth + Dividend Yield) / PE"""
        try:
            pe_ratio = self.info.get('trailingPE', None)
            growth_rate = self.info.get('earningsGrowth', None)
            dividend_yield = self.info.get('dividendYield', 0)

            if not pe_ratio or not growth_rate:
                return {
                    'value': 'N/A',
                    'status': 'Insufficient Data',
                    'description': 'PE or growth data not available'
                }

            # Convert to percentages
            growth_pct = growth_rate * 100 if growth_rate < 1 else growth_rate
            div_yield_pct = dividend_yield if dividend_yield else 0

            # Lynch Fair Value Formula
            lynch_value = (growth_pct + div_yield_pct) / pe_ratio

            # Interpretation
            if lynch_value > 1.5:
                status = 'Undervalued'
            elif lynch_value > 0.8:
                status = 'Fair'
            else:
                status = 'Overvalued'

            return {
                'value': f"{lynch_value:.2f}",
                'status': status,
                'description': f"({growth_pct:.1f}% + {div_yield_pct:.1f}%) ÷ {pe_ratio:.1f} = {lynch_value:.2f}",
                'raw_value': lynch_value
            }

        except Exception as e:
            return {
                'value': 'Error',
                'status': 'Error',
                'description': f'Error in Lynch Fair Value: {str(e)}'
            }

    # ================== WARREN BUFFETT METHODS ==================

    def calculate_buffett_criteria(self):
        """Warren Buffett's investment criteria checklist"""
        try:
            score = 0
            max_score = 10
            criteria_met = []
            criteria_failed = []

            # 1. ROE > 15% consistently
            roe = self.info.get('returnOnEquity', 0)
            if roe and roe > 0.15:
                score += 1
                criteria_met.append("High ROE (>15%)")
            else:
                criteria_failed.append("Low ROE (<15%)")

            # 2. Debt-to-Equity < 0.5
            debt_to_equity = self.info.get('debtToEquity', 100) / 100
            if debt_to_equity < 0.5:
                score += 1
                criteria_met.append("Low D/E (<0.5)")
            else:
                criteria_failed.append("High D/E (>0.5)")

            # 3. Current Ratio between 1.5 and 2.5
            current_ratio = self.info.get('currentRatio', 0)
            if 1.5 <= current_ratio <= 2.5:
                score += 1
                criteria_met.append("Good Current Ratio")
            else:
                criteria_failed.append("Poor Current Ratio")

            # 4. P/E < 15
            pe_ratio = self.info.get('trailingPE', 100)
            if pe_ratio and pe_ratio < 15:
                score += 1
                criteria_met.append("Low P/E (<15)")
            else:
                criteria_failed.append("High P/E (>15)")

            # 5. P/B < 1.5
            pb_ratio = self.info.get('priceToBook', 100)
            if pb_ratio and pb_ratio < 1.5:
                score += 1
                criteria_met.append("Low P/B (<1.5)")
            else:
                criteria_failed.append("High P/B (>1.5)")

            # 6. Profit margins > 20%
            profit_margins = self.info.get('profitMargins', 0)
            if profit_margins and profit_margins > 0.20:
                score += 1
                criteria_met.append("High Margins (>20%)")
            else:
                criteria_failed.append("Low Margins (<20%)")

            # 7. Revenue growth
            revenue_growth = self.info.get('revenueGrowth', 0)
            if revenue_growth and revenue_growth > 0.05:
                score += 1
                criteria_met.append("Revenue Growth (>5%)")
            else:
                criteria_failed.append("Low Revenue Growth")

            # 8. Free cash flow positive
            free_cash_flow = self.info.get('freeCashflow', 0)
            if free_cash_flow > 0:
                score += 1
                criteria_met.append("Positive FCF")
            else:
                criteria_failed.append("Negative FCF")

            # 9. Low capital expenditure needs
            operating_cf = self.info.get('operatingCashflow', 1)
            capex = abs(self.info.get('totalCashFromOperatingActivities', 0) - free_cash_flow)
            if operating_cf > 0 and capex / operating_cf < 0.3:
                score += 1
                criteria_met.append("Low CapEx needs")
            else:
                criteria_failed.append("High CapEx needs")

            # 10. Competitive moat (simplified - based on margins and ROE)
            if roe > 0.15 and profit_margins > 0.15:
                score += 1
                criteria_met.append("Potential Moat")
            else:
                criteria_failed.append("Weak Moat")

            # Determine status
            if score >= 8:
                status = 'Buffett-Quality'
            elif score >= 6:
                status = 'Good Quality'
            elif score >= 4:
                status = 'Average'
            else:
                status = 'Poor Quality'

            return {
                'value': f"{score}/{max_score}",
                'status': status,
                'description': f"Meets {score} of {max_score} Buffett criteria",
                'criteria_met': criteria_met,
                'criteria_failed': criteria_failed,
                'raw_score': score
            }

        except Exception as e:
            return {
                'value': 'Error',
                'status': 'Error',
                'description': f'Error in Buffett criteria: {str(e)}'
            }

    def calculate_buffett_intrinsic_value(self):
        """Buffett-style DCF with conservative assumptions"""
        try:
            # Get owner earnings (Buffett's version of free cash flow)
            net_income = self.info.get('netIncomeToCommon', 0)
            depreciation = self.info.get('totalCashFromOperatingActivities', 0) - self.info.get('freeCashflow', 0)
            capex_maintenance = depreciation * 0.8  # Assume 80% of depreciation is maintenance capex

            owner_earnings = net_income + depreciation - capex_maintenance

            if owner_earnings <= 0:
                return {
                    'value': 'N/A',
                    'status': 'Insufficient Data',
                    'description': 'Unable to calculate owner earnings'
                }

            shares_outstanding = self.info.get('sharesOutstanding', 1)

            # Conservative growth assumptions (Buffett-style)
            # Years 1-5: Historical growth rate or 5%, whichever is lower
            historical_growth = self.info.get('earningsGrowth', 0.05)
            growth_phase1 = min(0.05, historical_growth) if historical_growth > 0 else 0.03

            # Years 6-10: Fade to 3%
            growth_phase2 = 0.03

            # Terminal: 2.5%
            terminal_growth = 0.025

            # Discount rate: 10% (Buffett's hurdle rate)
            discount_rate = 0.10

            # Calculate DCF
            present_value = 0

            # Phase 1: Years 1-5
            for year in range(1, 6):
                future_earnings = owner_earnings * ((1 + growth_phase1) ** year)
                pv = future_earnings / ((1 + discount_rate) ** year)
                present_value += pv

            # Phase 2: Years 6-10
            for year in range(6, 11):
                future_earnings = owner_earnings * ((1 + growth_phase1) ** 5) * ((1 + growth_phase2) ** (year - 5))
                pv = future_earnings / ((1 + discount_rate) ** year)
                present_value += pv

            # Terminal value
            terminal_earnings = owner_earnings * ((1 + growth_phase1) ** 5) * ((1 + growth_phase2) ** 5) * (1 + terminal_growth)
            terminal_value = terminal_earnings / (discount_rate - terminal_growth)
            pv_terminal = terminal_value / ((1 + discount_rate) ** 10)

            total_value = present_value + pv_terminal
            intrinsic_value_per_share = total_value / shares_outstanding

            # Margin of safety
            current_price = self.info.get('currentPrice', 0)
            if current_price > 0:
                margin_of_safety = (intrinsic_value_per_share - current_price) / current_price * 100

                if margin_of_safety > 30:
                    status = 'Strong Buy'
                elif margin_of_safety > 15:
                    status = 'Buy'
                elif margin_of_safety > -10:
                    status = 'Fair'
                else:
                    status = 'Overvalued'
            else:
                status = 'Unknown'
                margin_of_safety = 0

            return {
                'value': f"${intrinsic_value_per_share:.2f}",
                'status': status,
                'description': f"Intrinsic: ${intrinsic_value_per_share:.2f}, Margin: {margin_of_safety:.1f}%",
                'raw_value': intrinsic_value_per_share,
                'margin_of_safety': margin_of_safety,
                'owner_earnings': owner_earnings
            }

        except Exception as e:
            return {
                'value': 'Error',
                'status': 'Error',
                'description': f'Error in Buffett intrinsic value: {str(e)}'
            }

    # ================== BENJAMIN GRAHAM METHODS ==================

    def calculate_graham_net_net(self):
        """Benjamin Graham's Net-Net Working Capital method"""
        try:
            # Get balance sheet data
            total_current_assets = self.info.get('totalCurrentAssets', 0)
            cash = self.info.get('totalCash', 0)
            receivables = self.info.get('totalCurrentAssets', 0) * 0.3  # Estimate receivables
            inventory = self.info.get('totalCurrentAssets', 0) * 0.2   # Estimate inventory

            # Conservative Graham formula
            # NNWC = Cash + (75% × Receivables) + (50% × Inventory) - Total Liabilities
            conservative_receivables = receivables * 0.75
            conservative_inventory = inventory * 0.50

            total_liabilities = self.info.get('totalLiab', 0)
            preferred_stock = self.info.get('preferredStock', 0)

            nnwc = cash + conservative_receivables + conservative_inventory - total_liabilities - preferred_stock

            shares_outstanding = self.info.get('sharesOutstanding', 1)
            nnwc_per_share = nnwc / shares_outstanding

            current_price = self.info.get('currentPrice', 0)

            # Graham's criteria: Buy if trading below 67% of NNWC
            graham_buy_price = nnwc_per_share * 0.67

            if current_price == 0:
                status = 'Unknown'
            elif current_price < graham_buy_price:
                status = 'Graham Buy'
            elif current_price < nnwc_per_share:
                status = 'Below NNWC'
            else:
                status = 'Above NNWC'

            # Additional Graham criteria
            eps_ttm = self.info.get('trailingEps', 0)
            earnings_positive = eps_ttm > 0

            if not earnings_positive and status == 'Graham Buy':
                status = 'Buy (No Earnings)'

            return {
                'value': f"${nnwc_per_share:.2f}",
                'status': status,
                'description': f"NNWC/share: ${nnwc_per_share:.2f}, Buy below: ${graham_buy_price:.2f}",
                'raw_value': nnwc_per_share,
                'buy_price': graham_buy_price,
                'total_nnwc': nnwc,
                'earnings_positive': earnings_positive
            }

        except Exception as e:
            return {
                'value': 'Error',
                'status': 'Error',
                'description': f'Error in Graham Net-Net: {str(e)}'
            }

    def calculate_graham_defensive_criteria(self):
        """Graham's Defensive Investor criteria"""
        try:
            score = 0
            max_score = 7
            criteria = []

            # 1. Adequate size (>$100M market cap)
            market_cap = self.info.get('marketCap', 0)
            if market_cap > 100e6:
                score += 1
                criteria.append("✓ Adequate size")
            else:
                criteria.append("✗ Too small")

            # 2. Strong financial condition (Current Ratio > 2)
            current_ratio = self.info.get('currentRatio', 0)
            if current_ratio > 2:
                score += 1
                criteria.append("✓ Strong finances")
            else:
                criteria.append("✗ Weak finances")

            # 3. Earnings stability (no losses in past 10 years - simplified)
            eps = self.info.get('trailingEps', 0)
            if eps > 0:
                score += 1
                criteria.append("✓ Positive earnings")
            else:
                criteria.append("✗ Negative earnings")

            # 4. Dividend record (10 years of dividends - simplified)
            dividend_rate = self.info.get('dividendRate', 0)
            if dividend_rate > 0:
                score += 1
                criteria.append("✓ Pays dividends")
            else:
                criteria.append("✗ No dividends")

            # 5. Earnings growth (33% in past 10 years - simplified to 3% annually)
            earnings_growth = self.info.get('earningsGrowth', 0)
            if earnings_growth > 0.03:
                score += 1
                criteria.append("✓ Earnings growth")
            else:
                criteria.append("✗ No earnings growth")

            # 6. Moderate P/E (not over 15 times average earnings)
            pe_ratio = self.info.get('trailingPE', 100)
            if pe_ratio and pe_ratio < 15:
                score += 1
                criteria.append("✓ Moderate P/E")
            else:
                criteria.append("✗ High P/E")

            # 7. Moderate P/B (not over 1.5 times book value)
            pb_ratio = self.info.get('priceToBook', 100)
            if pb_ratio and pb_ratio < 1.5:
                score += 1
                criteria.append("✓ Moderate P/B")
            else:
                criteria.append("✗ High P/B")

            # Graham's additional rule: P/E × P/B < 22.5
            if pe_ratio and pb_ratio and (pe_ratio * pb_ratio < 22.5):
                bonus = " (P/E×P/B✓)"
            else:
                bonus = " (P/E×P/B✗)"

            if score >= 6:
                status = 'Graham Defensive'
            elif score >= 4:
                status = 'Acceptable'
            else:
                status = 'Speculative'

            return {
                'value': f"{score}/{max_score}",
                'status': status + bonus,
                'description': f"Meets {score} of {max_score} defensive criteria",
                'criteria_list': criteria,
                'raw_score': score
            }

        except Exception as e:
            return {
                'value': 'Error',
                'status': 'Error',
                'description': f'Error in Graham Defensive: {str(e)}'
            }

    # ================== MAGIC FORMULA (JOEL GREENBLATT) ==================

    def calculate_magic_formula(self):
        """Joel Greenblatt's Magic Formula"""
        try:
            # 1. Earnings Yield = EBIT / Enterprise Value
            ebitda = self.info.get('ebitda', 0)
            market_cap = self.info.get('marketCap', 0)
            total_debt = self.info.get('totalDebt', 0)
            total_cash = self.info.get('totalCash', 0)

            enterprise_value = market_cap + total_debt - total_cash

            if enterprise_value <= 0 or ebitda <= 0:
                return {
                    'value': 'N/A',
                    'status': 'Insufficient Data',
                    'description': 'EBITDA or Enterprise Value not available'
                }

            earnings_yield = ebitda / enterprise_value * 100

            # 2. Return on Capital = EBIT / (Net Fixed Assets + Working Capital)
            total_assets = self.info.get('totalAssets', 0)
            total_current_liabilities = self.info.get('totalCurrentLiabilities', 0)
            total_current_assets = self.info.get('totalCurrentAssets', 0)

            working_capital = total_current_assets - total_current_liabilities
            net_fixed_assets = total_assets - total_current_assets
            invested_capital = net_fixed_assets + working_capital

            if invested_capital <= 0:
                return_on_capital = 0
            else:
                return_on_capital = ebitda / invested_capital * 100

            # Magic Formula Interpretation
            # High earnings yield (cheap) + High return on capital (good business)
            if earnings_yield > 10 and return_on_capital > 20:
                status = 'Magic Formula Buy'
            elif earnings_yield > 7 and return_on_capital > 15:
                status = 'Good Candidate'
            elif earnings_yield > 5 and return_on_capital > 10:
                status = 'Fair'
            else:
                status = 'Avoid'

            # Combined score (simple average of percentile ranks)
            magic_score = (min(earnings_yield, 25) + min(return_on_capital, 50)) / 2

            return {
                'value': f"{magic_score:.1f}",
                'status': status,
                'description': f"EY: {earnings_yield:.1f}%, ROC: {return_on_capital:.1f}%",
                'earnings_yield': earnings_yield,
                'return_on_capital': return_on_capital,
                'raw_value': magic_score
            }

        except Exception as e:
            return {
                'value': 'Error',
                'status': 'Error',
                'description': f'Error in Magic Formula: {str(e)}'
            }

    # ================== HEDGE FUND METHODS ==================

    def calculate_momentum_factor(self):
        """Hedge Fund style momentum factor"""
        try:
            if self.history.empty or len(self.history) < 60:
                return {
                    'value': 'N/A',
                    'status': 'Insufficient Data',
                    'description': 'Need at least 60 days of price history'
                }

            # Multiple momentum timeframes
            returns_1m = self.history['Close'].pct_change(20).iloc[-1] if len(self.history) >= 20 else 0
            returns_3m = self.history['Close'].pct_change(60).iloc[-1] if len(self.history) >= 60 else 0
            returns_6m = self.history['Close'].pct_change(120).iloc[-1] if len(self.history) >= 120 else 0
            returns_12m = self.history['Close'].pct_change(252).iloc[-1] if len(self.history) >= 252 else 0

            # Weight recent performance more heavily
            momentum_score = (returns_1m * 0.4 + returns_3m * 0.3 + returns_6m * 0.2 + returns_12m * 0.1) * 100

            # Risk-adjusted momentum (Sharpe-like)
            daily_returns = self.history['Close'].pct_change().dropna()
            volatility = daily_returns.std() * np.sqrt(252) * 100

            if volatility > 0:
                risk_adjusted_momentum = momentum_score / volatility
            else:
                risk_adjusted_momentum = 0

            # Status determination
            if momentum_score > 20:
                status = 'Strong Momentum'
            elif momentum_score > 10:
                status = 'Positive Momentum'
            elif momentum_score > -5:
                status = 'Neutral'
            elif momentum_score > -20:
                status = 'Negative Momentum'
            else:
                status = 'Strong Decline'

            return {
                'value': f"{momentum_score:.1f}%",
                'status': status,
                'description': f"1M: {returns_1m*100:.1f}%, 3M: {returns_3m*100:.1f}%, Vol: {volatility:.1f}%",
                'raw_value': momentum_score,
                'risk_adjusted': risk_adjusted_momentum,
                'volatility': volatility
            }

        except Exception as e:
            return {
                'value': 'Error',
                'status': 'Error',
                'description': f'Error in momentum calculation: {str(e)}'
            }

    def calculate_quality_factor(self):
        """Hedge Fund quality factor"""
        try:
            quality_score = 0
            max_score = 100
            components = []

            # Profitability (40 points)
            roe = self.info.get('returnOnEquity', 0)
            roa = self.info.get('returnOnAssets', 0)
            profit_margins = self.info.get('profitMargins', 0)

            if roe > 0.15:
                quality_score += 15
                components.append("High ROE")
            elif roe > 0.10:
                quality_score += 10
                components.append("Good ROE")

            if roa > 0.05:
                quality_score += 10
                components.append("High ROA")
            elif roa > 0.02:
                quality_score += 5
                components.append("Good ROA")

            if profit_margins > 0.15:
                quality_score += 15
                components.append("High Margins")
            elif profit_margins > 0.10:
                quality_score += 10
                components.append("Good Margins")

            # Financial Strength (30 points)
            debt_to_equity = self.info.get('debtToEquity', 100)
            current_ratio = self.info.get('currentRatio', 0)

            if debt_to_equity < 30:
                quality_score += 15
                components.append("Low Debt")
            elif debt_to_equity < 60:
                quality_score += 10
                components.append("Moderate Debt")

            if current_ratio > 2:
                quality_score += 15
                components.append("Strong Liquidity")
            elif current_ratio > 1.5:
                quality_score += 10
                components.append("Good Liquidity")

            # Growth & Stability (30 points)
            revenue_growth = self.info.get('revenueGrowth', 0)
            earnings_growth = self.info.get('earningsGrowth', 0)

            if revenue_growth > 0.10:
                quality_score += 15
                components.append("High Revenue Growth")
            elif revenue_growth > 0.05:
                quality_score += 10
                components.append("Good Revenue Growth")

            if earnings_growth > 0.10:
                quality_score += 15
                components.append("High EPS Growth")
            elif earnings_growth > 0.05:
                quality_score += 10
                components.append("Good EPS Growth")

            # Status
            if quality_score >= 80:
                status = 'High Quality'
            elif quality_score >= 60:
                status = 'Good Quality'
            elif quality_score >= 40:
                status = 'Average Quality'
            else:
                status = 'Low Quality'

            return {
                'value': f"{quality_score}/100",
                'status': status,
                'description': f"Score: {quality_score}, Components: {', '.join(components[:3])}",
                'raw_value': quality_score,
                'components': components
            }

        except Exception as e:
            return {
                'value': 'Error',
                'status': 'Error',
                'description': f'Error in quality calculation: {str(e)}'
            }

    def calculate_mean_reversion(self):
        """Mean reversion analysis (contrarian hedge fund strategy)"""
        try:
            if self.history.empty or len(self.history) < 100:
                return {
                    'value': 'N/A',
                    'status': 'Insufficient Data',
                    'description': 'Need at least 100 days of price history'
                }

            current_price = self.history['Close'].iloc[-1]

            # Multiple moving averages
            ma_20 = self.history['Close'].rolling(20).mean().iloc[-1]
            ma_50 = self.history['Close'].rolling(50).mean().iloc[-1] if len(self.history) >= 50 else ma_20
            ma_100 = self.history['Close'].rolling(100).mean().iloc[-1] if len(self.history) >= 100 else ma_50

            # Distance from means
            distance_20 = (current_price - ma_20) / ma_20 * 100
            distance_50 = (current_price - ma_50) / ma_50 * 100
            distance_100 = (current_price - ma_100) / ma_100 * 100

            # Bollinger Bands analysis
            bb_upper = ma_20 + (self.history['Close'].rolling(20).std().iloc[-1] * 2)
            bb_lower = ma_20 - (self.history['Close'].rolling(20).std().iloc[-1] * 2)
            bb_position = (current_price - bb_lower) / (bb_upper - bb_lower)

            # Mean reversion score (negative means oversold, positive means overbought)
            mean_reversion_score = (distance_20 * 0.5 + distance_50 * 0.3 + distance_100 * 0.2)

            # Status based on extremes (mean reversion opportunities)
            if mean_reversion_score < -15 or bb_position < 0.2:
                status = 'Oversold (Buy)'
            elif mean_reversion_score < -7:
                status = 'Potentially Oversold'
            elif mean_reversion_score > 15 or bb_position > 0.8:
                status = 'Overbought (Sell)'
            elif mean_reversion_score > 7:
                status = 'Potentially Overbought'
            else:
                status = 'Neutral'

            return {
                'value': f"{mean_reversion_score:.1f}%",
                'status': status,
                'description': f"vs MA20: {distance_20:.1f}%, BB Position: {bb_position:.2f}",
                'raw_value': mean_reversion_score,
                'bb_position': bb_position,
                'distances': {'20d': distance_20, '50d': distance_50, '100d': distance_100}
            }

        except Exception as e:
            return {
                'value': 'Error',
                'status': 'Error',
                'description': f'Error in mean reversion: {str(e)}'
            }

    def calculate_multi_factor_score(self):
        """Hedge fund style multi-factor model"""
        try:
            factors = {}
            weights = {
                'value': 0.25,
                'quality': 0.25,
                'momentum': 0.25,
                'low_volatility': 0.25
            }

            # Value Factor (0-100)
            pe_ratio = self.info.get('trailingPE', None)
            pb_ratio = self.info.get('priceToBook', None)

            value_score = 50  # neutral start
            if pe_ratio:
                if pe_ratio < 10:
                    value_score += 25
                elif pe_ratio < 15:
                    value_score += 15
                elif pe_ratio > 25:
                    value_score -= 20

            if pb_ratio:
                if pb_ratio < 1:
                    value_score += 25
                elif pb_ratio < 1.5:
                    value_score += 15
                elif pb_ratio > 3:
                    value_score -= 20

            factors['value'] = max(0, min(100, value_score))

            # Quality Factor (from previous method)
            quality_result = self.calculate_quality_factor()
            factors['quality'] = quality_result['raw_value'] if quality_result['raw_value'] != 'Error' else 50

            # Momentum Factor (normalized to 0-100)
            momentum_result = self.calculate_momentum_factor()
            if momentum_result['raw_value'] != 'Error':
                # Convert momentum percentage to 0-100 scale
                momentum_pct = momentum_result['raw_value']
                factors['momentum'] = max(0, min(100, 50 + momentum_pct * 2))  # Scale momentum
            else:
                factors['momentum'] = 50

            # Low Volatility Factor (inverted - low vol gets high score)
            if not self.history.empty and len(self.history) > 20:
                daily_returns = self.history['Close'].pct_change().dropna()
                volatility = daily_returns.std() * np.sqrt(252) * 100

                # Invert volatility - lower vol = higher score
                if volatility < 15:
                    factors['low_volatility'] = 90
                elif volatility < 25:
                    factors['low_volatility'] = 70
                elif volatility < 35:
                    factors['low_volatility'] = 50
                elif volatility < 50:
                    factors['low_volatility'] = 30
                else:
                    factors['low_volatility'] = 10
            else:
                factors['low_volatility'] = 50

            # Calculate weighted score
            multi_factor_score = sum(factors[factor] * weights[factor] for factor in factors)

            # Status
            if multi_factor_score >= 75:
                status = 'Strong Buy'
            elif multi_factor_score >= 60:
                status = 'Buy'
            elif multi_factor_score >= 40:
                status = 'Hold'
            elif multi_factor_score >= 25:
                status = 'Sell'
            else:
                status = 'Strong Sell'

            return {
                'value': f"{multi_factor_score:.1f}",
                'status': status,
                'description': f"V:{factors['value']:.0f} Q:{factors['quality']:.0f} M:{factors['momentum']:.0f} LV:{factors['low_volatility']:.0f}",
                'raw_value': multi_factor_score,
                'factor_scores': factors
            }

        except Exception as e:
            return {
                'value': 'Error',
                'status': 'Error',
                'description': f'Error in multi-factor score: {str(e)}'
            }
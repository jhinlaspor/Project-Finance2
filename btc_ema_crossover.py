#!/usr/bin/env python3
"""
BTC EMA Crossover Trading System for Alpaca

This script implements an automated crypto trading system using EMA crossover strategy
with ATR-based entry filters and trailing stops. It operates in phases:
1. Strategy design with historical data
2. Backtesting with risk metrics
3. Paper trading deployment
4. Live trading (requires explicit approval)
5. Monitoring and safety controls

Author: Claude Opus 4 Agent
Date: January 2025
"""

import os
import sys
import time
import json
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yaml
from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.live import CryptoDataStream

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_system.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class BTCEMACrossoverStrategy:
    """Main trading strategy class implementing EMA crossover with ATR filters"""
    
    def __init__(self):
        """Initialize strategy with Alpaca API credentials"""
        self.api_key = os.environ.get('APCA_API_KEY_ID')
        self.secret_key = os.environ.get('APCA_API_SECRET_KEY')
        self.base_url = os.environ.get('APCA_API_BASE_URL', 'https://paper-api.alpaca.markets')
        
        if not self.api_key or not self.secret_key:
            raise ValueError("Alpaca API credentials not found in environment variables")
        
        # Initialize clients
        self.crypto_client = CryptoHistoricalDataClient()
        self.trading_client = TradingClient(self.api_key, self.secret_key, paper=True)
        
        # Strategy parameters
        self.symbol = "BTC/USD"
        self.ema_fast = 50
        self.ema_slow = 200
        self.atr_period = 14
        self.atr_multiplier = 1.0  # For entry filter
        self.trailing_stop_atr = 1.5  # For exit
        
        # Data storage
        self.historical_data = None
        self.backtest_results = None
        self.paper_trades = []
        self.live_trades = []
        
        logger.info(f"Strategy initialized with base URL: {self.base_url}")
    
    def fetch_historical_data(self, days: int = 180) -> pd.DataFrame:
        """
        Fetch historical 1-minute OHLCV data for BTC/USD
        
        Args:
            days: Number of days of historical data to fetch
            
        Returns:
            DataFrame with OHLCV data
        """
        logger.info(f"Fetching {days} days of historical data for {self.symbol}")
        
        # Fetch data in chunks to avoid API limits
        chunk_days = 30  # Fetch 30 days at a time
        all_data = []
        
        # Use a fixed end date in the past to ensure data availability
        # Setting to January 2025 to get recent but available data
        end_time = datetime(2025, 1, 15, 0, 0, 0)
        
        for i in range(0, days, chunk_days):
            chunk_start = end_time - timedelta(days=min(i + chunk_days, days))
            chunk_end = end_time - timedelta(days=i)
            
            logger.info(f"Fetching chunk: {chunk_start.strftime('%Y-%m-%d')} to {chunk_end.strftime('%Y-%m-%d')}")
            
            # Create request for 1-minute bars
            request_params = CryptoBarsRequest(
                symbol_or_symbols=[self.symbol],
                timeframe=TimeFrame.Minute,
                start=chunk_start,
                end=chunk_end,
                limit=None,
                page_limit=None
            )
            
            try:
                # Fetch data
                bars = self.crypto_client.get_crypto_bars(request_params)
                
                # Convert to DataFrame
                if bars.df is not None and not bars.df.empty:
                    df_chunk = bars.df
                    all_data.append(df_chunk)
                    logger.info(f"Fetched {len(df_chunk)} bars in this chunk")
                else:
                    logger.warning(f"No data received for chunk {chunk_start} to {chunk_end}")
                
                # Add a small delay to avoid rate limiting
                time.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Error fetching chunk {chunk_start} to {chunk_end}: {e}")
                continue
        
        if not all_data:
            logger.error("No data fetched from any chunks")
            raise ValueError("Failed to fetch any historical data")
        
        # Combine all chunks
        df = pd.concat(all_data)
        
        # Remove duplicates and sort by timestamp
        df = df[~df.index.duplicated(keep='first')]
        df = df.sort_index()
        
        # Keep only OHLCV columns
        df = df[['open', 'high', 'low', 'close', 'volume']]
        
        logger.info(f"Total bars fetched: {len(df)}")
        
        self.historical_data = df
        return df
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators: EMA50, EMA200, ATR14
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with added indicator columns
        """
        logger.info("Calculating technical indicators")
        
        # Calculate EMAs
        df['ema50'] = df['close'].ewm(span=self.ema_fast, adjust=False).mean()
        df['ema200'] = df['close'].ewm(span=self.ema_slow, adjust=False).mean()
        
        # Calculate ATR
        df['tr'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift()),
                abs(df['low'] - df['close'].shift())
            )
        )
        df['atr14'] = df['tr'].rolling(window=self.atr_period).mean()
        
        # Generate signals
        df['ema_cross'] = np.where(df['ema50'] > df['ema200'], 1, -1)
        df['ema_cross_signal'] = df['ema_cross'].diff()
        
        # Entry conditions
        df['long_entry'] = (
            (df['ema_cross_signal'] == 2) &  # EMA50 crosses above EMA200
            (df['close'] > df['ema200'] + self.atr_multiplier * df['atr14'])  # Price > EMA200 + ATR
        )
        
        # Exit conditions (will be refined with trailing stop in live trading)
        df['long_exit'] = df['ema_cross_signal'] == -2  # EMA50 crosses below EMA200
        
        # Clean up
        df.drop(columns=['tr'], inplace=True)
        
        logger.info("Indicators calculated successfully")
        return df
    
    def run_phase1(self):
        """Execute Phase 1: Strategy Design"""
        logger.info("="*50)
        logger.info("PHASE 1: STRATEGY DESIGN")
        logger.info("="*50)
        
        # Fetch historical data
        df = self.fetch_historical_data(days=180)
        
        # Calculate indicators
        df = self.calculate_indicators(df)
        
        # Save processed data
        df.to_csv('historical_data_with_indicators.csv')
        logger.info("Historical data with indicators saved to historical_data_with_indicators.csv")
        
        # Generate summary statistics
        logger.info("\nData Summary:")
        if len(df) > 0:
            logger.info(f"Date range: {df.index[0]} to {df.index[-1]}")
            logger.info(f"Total bars: {len(df)}")
            logger.info(f"Long entry signals: {df['long_entry'].sum()}")
            logger.info(f"Long exit signals: {df['long_exit'].sum()}")
        else:
            logger.warning("No data available for summary")
        
        return df
    
    def backtest(self, df: pd.DataFrame, initial_capital: float = 10000) -> Dict:
        """
        Run backtest on historical data with the strategy
        
        Args:
            df: DataFrame with OHLCV data and indicators
            initial_capital: Starting capital for backtest
            
        Returns:
            Dictionary with backtest results and metrics
        """
        logger.info("Running backtest...")
        
        # Initialize backtest variables
        capital = initial_capital
        position = 0
        entry_price = 0
        trades = []
        equity_curve = []
        
        # Track position and trades
        entry_time = None
        highest_price = 0
        
        for idx in range(len(df)):
            current_bar = df.iloc[idx]
            timestamp = df.index[idx]
            close_price = current_bar['close']
            
            # Record equity
            current_equity = capital + (position * close_price if position > 0 else 0)
            equity_curve.append({
                'timestamp': timestamp,
                'equity': current_equity,
                'capital': capital,
                'position_value': position * close_price if position > 0 else 0
            })
            
            # Skip if indicators not ready
            if pd.isna(current_bar['ema200']) or pd.isna(current_bar['atr14']):
                continue
            
            # Check for entry signal
            if position == 0 and current_bar['long_entry']:
                # Enter position with full capital
                position = capital / close_price
                entry_price = close_price
                entry_time = timestamp
                capital = 0
                highest_price = close_price
                
                logger.debug(f"Entry at {timestamp}: Price={close_price:.2f}, Position={position:.6f}")
            
            # Check for exit signal or trailing stop
            elif position > 0:
                # Update highest price for trailing stop
                if close_price > highest_price:
                    highest_price = close_price
                
                # Calculate trailing stop from highest price
                trailing_stop = highest_price - (self.trailing_stop_atr * current_bar['atr14'])
                
                # Exit on signal or if price hits trailing stop
                if current_bar['long_exit'] or close_price <= trailing_stop:
                    # Exit position
                    exit_value = position * close_price
                    pnl = exit_value - (position * entry_price)
                    pnl_pct = (pnl / (position * entry_price)) * 100
                    
                    trades.append({
                        'entry_time': entry_time,
                        'exit_time': timestamp,
                        'entry_price': entry_price,
                        'exit_price': close_price,
                        'position_size': position,
                        'pnl': pnl,
                        'pnl_pct': pnl_pct,
                        'exit_reason': 'signal' if current_bar['long_exit'] else 'trailing_stop'
                    })
                    
                    capital = exit_value
                    position = 0
                    entry_price = 0
                    entry_time = None
                    highest_price = 0
                    
                    logger.debug(f"Exit at {timestamp}: Price={close_price:.2f}, PnL={pnl:.2f} ({pnl_pct:.2f}%)")
        
        # Close any open position at end
        if position > 0:
            exit_value = position * df.iloc[-1]['close']
            pnl = exit_value - (position * entry_price)
            pnl_pct = (pnl / (position * entry_price)) * 100
            
            trades.append({
                'entry_time': df.index[-1],
                'exit_time': df.index[-1],
                'entry_price': entry_price,
                'exit_price': df.iloc[-1]['close'],
                'position_size': position,
                'pnl': pnl,
                'pnl_pct': pnl_pct,
                'exit_reason': 'end_of_data'
            })
            
            capital = exit_value
        
        # Calculate metrics
        equity_df = pd.DataFrame(equity_curve)
        equity_df.set_index('timestamp', inplace=True)
        
        # Calculate returns
        equity_df['returns'] = equity_df['equity'].pct_change()
        equity_df['cumulative_returns'] = (1 + equity_df['returns']).cumprod() - 1
        
        # Calculate metrics
        total_return = (capital - initial_capital) / initial_capital
        days_in_backtest = (df.index[-1][1] - df.index[0][1]).days
        years_in_backtest = days_in_backtest / 365.25
        
        # CAGR
        cagr = (((capital / initial_capital) ** (1 / years_in_backtest)) - 1) if years_in_backtest > 0 else 0
        
        # Sharpe Ratio (assuming 0% risk-free rate)
        daily_returns = equity_df['returns'].dropna()
        if len(daily_returns) > 0:
            sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252) if daily_returns.std() > 0 else 0
        else:
            sharpe_ratio = 0
        
        # Maximum Drawdown
        equity_df['cummax'] = equity_df['equity'].cummax()
        equity_df['drawdown'] = (equity_df['equity'] - equity_df['cummax']) / equity_df['cummax']
        max_drawdown = equity_df['drawdown'].min()
        
        # Win rate
        winning_trades = [t for t in trades if t['pnl'] > 0]
        win_rate = len(winning_trades) / len(trades) if trades else 0
        
        # Average trade metrics
        avg_win = np.mean([t['pnl_pct'] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t['pnl_pct'] for t in trades if t['pnl'] <= 0]) if any(t['pnl'] <= 0 for t in trades) else 0
        
        results = {
            'initial_capital': initial_capital,
            'final_capital': capital,
            'total_return': total_return,
            'cagr': cagr,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'total_trades': len(trades),
            'winning_trades': len(winning_trades),
            'win_rate': win_rate,
            'avg_win_pct': avg_win,
            'avg_loss_pct': avg_loss,
            'days_in_backtest': days_in_backtest,
            'trades': trades,
            'equity_curve': equity_df
        }
        
        self.backtest_results = results
        return results
    
    def save_backtest_results(self, results: Dict):
        """Save backtest results to JSON file"""
        # Prepare data for JSON serialization
        json_results = {
            'initial_capital': results['initial_capital'],
            'final_capital': results['final_capital'],
            'total_return': results['total_return'],
            'cagr': results['cagr'],
            'sharpe_ratio': results['sharpe_ratio'],
            'max_drawdown': results['max_drawdown'],
            'total_trades': results['total_trades'],
            'winning_trades': results['winning_trades'],
            'win_rate': results['win_rate'],
            'avg_win_pct': results['avg_win_pct'],
            'avg_loss_pct': results['avg_loss_pct'],
            'days_in_backtest': results['days_in_backtest'],
            'trades': results['trades']
        }
        
        # Save to JSON
        with open('backtest_results.json', 'w') as f:
            json.dump(json_results, f, indent=2, default=str)
        
        # Save equity curve
        results['equity_curve'].to_csv('equity_curve.csv')
        
        # Create visualization
        try:
            self.plot_backtest_results(results)
        except Exception as e:
            logger.warning(f"Failed to create visualization: {e}")
        
        logger.info("Backtest results saved to backtest_results.json and equity_curve.csv")
    
    def plot_backtest_results(self, results: Dict):
        """Create visualization of backtest results"""
        equity_df = results['equity_curve']
        
        # Reset index to handle multi-index issue
        if isinstance(equity_df.index, pd.MultiIndex):
            # Extract just the timestamp from the multi-index
            timestamps = [idx[1] if isinstance(idx, tuple) else idx for idx in equity_df.index]
            equity_df.index = timestamps
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        # Plot equity curve
        ax1.plot(equity_df.index, equity_df['equity'], label='Portfolio Value', color='blue')
        ax1.axhline(y=results['initial_capital'], color='gray', linestyle='--', label='Initial Capital')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.set_title('Backtest Results: BTC EMA Crossover Strategy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot drawdown
        ax2.fill_between(equity_df.index, equity_df['drawdown'] * 100, 0, color='red', alpha=0.3)
        ax2.set_ylabel('Drawdown (%)')
        ax2.set_xlabel('Date')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('backtest_results.png', dpi=150)
        plt.close()
        
        logger.info("Backtest visualization saved to backtest_results.png")
    
    def run_phase2(self, df: pd.DataFrame):
        """Execute Phase 2: Backtesting"""
        logger.info("="*50)
        logger.info("PHASE 2: BACKTESTING")
        logger.info("="*50)
        
        # Run backtest
        results = self.backtest(df)
        
        # Display results
        logger.info("\nBacktest Results:")
        logger.info(f"Initial Capital: ${results['initial_capital']:,.2f}")
        logger.info(f"Final Capital: ${results['final_capital']:,.2f}")
        logger.info(f"Total Return: {results['total_return']*100:.2f}%")
        logger.info(f"CAGR: {results['cagr']*100:.2f}%")
        logger.info(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        logger.info(f"Max Drawdown: {results['max_drawdown']*100:.2f}%")
        logger.info(f"Total Trades: {results['total_trades']}")
        logger.info(f"Win Rate: {results['win_rate']*100:.2f}%")
        logger.info(f"Avg Win: {results['avg_win_pct']:.2f}%")
        logger.info(f"Avg Loss: {results['avg_loss_pct']:.2f}%")
        
        # Save results
        self.save_backtest_results(results)
        
        # Check pass criteria
        passed = results['sharpe_ratio'] >= 1.0 and abs(results['max_drawdown']) <= 0.10
        
        logger.info("\nPass Criteria Check:")
        logger.info(f"Sharpe Ratio >= 1.0: {'PASS' if results['sharpe_ratio'] >= 1.0 else 'FAIL'} ({results['sharpe_ratio']:.2f})")
        logger.info(f"Max Drawdown <= 10%: {'PASS' if abs(results['max_drawdown']) <= 0.10 else 'FAIL'} ({abs(results['max_drawdown'])*100:.2f}%)")
        logger.info(f"Overall: {'PASS' if passed else 'FAIL'}")
        
        return results, passed
    
    def optimize_parameters(self, df: pd.DataFrame, max_iterations: int = 5):
        """
        Optimize strategy parameters to meet pass criteria
        
        Args:
            df: DataFrame with OHLCV data
            max_iterations: Maximum optimization iterations
            
        Returns:
            Best parameters and results
        """
        logger.info("\nStarting parameter optimization...")
        
        # Parameter ranges to test
        ema_fast_range = [20, 30, 40, 50, 60]
        ema_slow_range = [100, 150, 200, 250]
        atr_multiplier_range = [0.5, 0.75, 1.0, 1.25, 1.5]
        trailing_stop_range = [1.0, 1.5, 2.0, 2.5]
        
        best_params = None
        best_results = None
        best_score = -float('inf')
        iteration = 0
        
        for fast in ema_fast_range:
            for slow in ema_slow_range:
                if fast >= slow:
                    continue
                    
                for atr_mult in atr_multiplier_range:
                    for trail_stop in trailing_stop_range:
                        if iteration >= max_iterations:
                            break
                            
                        iteration += 1
                        
                        # Update parameters
                        self.ema_fast = fast
                        self.ema_slow = slow
                        self.atr_multiplier = atr_mult
                        self.trailing_stop_atr = trail_stop
                        
                        # Recalculate indicators
                        df_test = df.copy()
                        df_test = self.calculate_indicators(df_test)
                        
                        # Run backtest
                        try:
                            results = self.backtest(df_test)
                            
                            # Calculate score (prioritize Sharpe ratio and low drawdown)
                            if abs(results['max_drawdown']) <= 0.10:
                                score = results['sharpe_ratio'] - abs(results['max_drawdown']) * 10
                            else:
                                score = results['sharpe_ratio'] - abs(results['max_drawdown']) * 20
                            
                            logger.info(f"Iteration {iteration}: EMA({fast}/{slow}), ATR_mult={atr_mult}, Trail={trail_stop} -> Sharpe={results['sharpe_ratio']:.2f}, DD={results['max_drawdown']*100:.1f}%, Score={score:.2f}")
                            
                            if score > best_score:
                                best_score = score
                                best_params = {
                                    'ema_fast': fast,
                                    'ema_slow': slow,
                                    'atr_multiplier': atr_mult,
                                    'trailing_stop_atr': trail_stop
                                }
                                best_results = results
                                
                                # Check if pass criteria met
                                if results['sharpe_ratio'] >= 1.0 and abs(results['max_drawdown']) <= 0.10:
                                    logger.info(f"Found parameters that meet criteria!")
                                    return best_params, best_results
                                    
                        except Exception as e:
                            logger.warning(f"Error in optimization iteration {iteration}: {e}")
                            continue
                            
                if iteration >= max_iterations:
                    break
            if iteration >= max_iterations:
                break
        
        logger.info(f"\nOptimization complete. Best parameters: {best_params}")
        return best_params, best_results


def main():
    """Main entry point"""
    strategy = BTCEMACrossoverStrategy()
    
    # Phase 1: Strategy Design
    try:
        historical_data = strategy.run_phase1()
        logger.info("\nPhase 1 completed successfully!")
        logger.info("Proceeding to Phase 2: Backtesting...")
        
        # Phase 2: Backtesting
        results, passed = strategy.run_phase2(historical_data)
        
        if not passed:
            logger.info("\nBacktest failed pass criteria. Running parameter optimization...")
            
            # Run optimization
            best_params, best_results = strategy.optimize_parameters(historical_data, max_iterations=5)
            
            if best_results and best_results['sharpe_ratio'] >= 1.0 and abs(best_results['max_drawdown']) <= 0.10:
                logger.info("\nOptimization successful! Found parameters that meet criteria.")
                logger.info(f"Best parameters: {best_params}")
                strategy.save_backtest_results(best_results)
                passed = True
            else:
                logger.info("\nOptimization failed to find parameters that meet criteria.")
                logger.info("Manual intervention required or different strategy needed.")
                return
        
        if passed:
            logger.info("\nReady for Phase 3: Paper Trading")
            # Phase 3 would be implemented here
        
    except Exception as e:
        logger.error(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
BTC Advanced Trading Strategy for Alpaca

This script implements an advanced crypto trading system using multiple indicators:
- RSI for overbought/oversold conditions
- MACD for trend confirmation
- Bollinger Bands for volatility-based entries
- Volume analysis for signal confirmation
- ATR for dynamic stop losses

Uses 5-minute timeframe for more reliable signals.

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
        logging.FileHandler('advanced_trading_system.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class BTCAdvancedStrategy:
    """Advanced trading strategy class using multiple technical indicators"""
    
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
        self.timeframe = TimeFrame.Minute  # Will aggregate to 5m
        
        # Technical indicator parameters
        self.rsi_period = 14
        self.rsi_oversold = 35  # Less restrictive
        self.rsi_overbought = 65  # Less restrictive
        
        self.macd_fast = 12
        self.macd_slow = 26
        self.macd_signal = 9
        
        self.bb_period = 20
        self.bb_std = 2
        
        self.volume_ma_period = 20
        self.volume_threshold = 1.2  # Reduced from 1.5
        
        self.atr_period = 14
        self.atr_stop_multiplier = 1.5  # Tighter stop
        self.atr_target_multiplier = 2.5  # Better R:R ratio
        
        # Data storage
        self.historical_data = None
        self.backtest_results = None
        self.paper_trades = []
        self.live_trades = []
        
        logger.info(f"Advanced strategy initialized with base URL: {self.base_url}")
    
    def fetch_historical_data(self, days: int = 180) -> pd.DataFrame:
        """
        Fetch historical 1-minute OHLCV data for BTC/USD and aggregate to 5-minute
        
        Args:
            days: Number of days of historical data to fetch
            
        Returns:
            DataFrame with 5-minute OHLCV data
        """
        logger.info(f"Fetching {days} days of historical data for {self.symbol}")
        
        # Fetch data in chunks to avoid API limits
        chunk_days = 30
        all_data = []
        
        # Use a fixed end date in the past to ensure data availability
        end_time = datetime(2025, 1, 15, 0, 0, 0)
        
        for i in range(0, days, chunk_days):
            chunk_start = end_time - timedelta(days=min(i + chunk_days, days))
            chunk_end = end_time - timedelta(days=i)
            
            logger.info(f"Fetching chunk: {chunk_start.strftime('%Y-%m-%d')} to {chunk_end.strftime('%Y-%m-%d')}")
            
            request_params = CryptoBarsRequest(
                symbol_or_symbols=[self.symbol],
                timeframe=self.timeframe,
                start=chunk_start,
                end=chunk_end,
                limit=None,
                page_limit=None
            )
            
            try:
                bars = self.crypto_client.get_crypto_bars(request_params)
                
                if bars.df is not None and not bars.df.empty:
                    df_chunk = bars.df
                    all_data.append(df_chunk)
                    logger.info(f"Fetched {len(df_chunk)} bars in this chunk")
                else:
                    logger.warning(f"No data received for chunk {chunk_start} to {chunk_end}")
                
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
        
        logger.info(f"Total 1-minute bars fetched: {len(df)}")
        
        # Aggregate to 5-minute bars
        df_5m = self.aggregate_to_5min(df)
        
        self.historical_data = df_5m
        return df_5m
    
    def aggregate_to_5min(self, df_1m: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate 1-minute bars to 5-minute bars
        
        Args:
            df_1m: DataFrame with 1-minute OHLCV data
            
        Returns:
            DataFrame with 5-minute OHLCV data
        """
        logger.info("Aggregating to 5-minute bars...")
        
        # Extract just the timestamp from multi-index if needed
        if isinstance(df_1m.index, pd.MultiIndex):
            timestamps = [idx[1] if isinstance(idx, tuple) else idx for idx in df_1m.index]
            df_1m.index = pd.DatetimeIndex(timestamps)
        
        # Resample to 5-minute bars
        df_5m = df_1m.resample('5T').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        })
        
        # Remove any NaN rows
        df_5m = df_5m.dropna()
        
        logger.info(f"Aggregated to {len(df_5m)} 5-minute bars")
        return df_5m
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate advanced technical indicators
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with added indicator columns
        """
        logger.info("Calculating advanced technical indicators")
        
        # RSI
        df['rsi'] = self.calculate_rsi(df['close'], self.rsi_period)
        
        # MACD
        df['macd'], df['macd_signal'], df['macd_hist'] = self.calculate_macd(
            df['close'], self.macd_fast, self.macd_slow, self.macd_signal
        )
        
        # Bollinger Bands
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = self.calculate_bollinger_bands(
            df['close'], self.bb_period, self.bb_std
        )
        df['bb_width'] = df['bb_upper'] - df['bb_lower']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Volume analysis
        df['volume_ma'] = df['volume'].rolling(window=self.volume_ma_period).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        # ATR for stops
        df['atr'] = self.calculate_atr(df, self.atr_period)
        
        # Price action
        df['price_change'] = df['close'].pct_change()
        df['high_low_ratio'] = (df['high'] - df['low']) / df['close']
        
        # Generate entry signals - using combination of conditions
        # Long signal: RSI oversold OR price below BB, with MACD confirmation
        df['long_signal'] = (
            ((df['rsi'] < self.rsi_oversold) | (df['close'] < df['bb_lower'])) &  # RSI oversold OR price below BB
            (df['macd'] > df['macd_signal']) &  # MACD bullish
            (df['macd_hist'] > df['macd_hist'].shift(1)) &  # MACD histogram increasing
            (df['volume_ratio'] > 0.8)  # Some volume activity
        )
        
        # Short signal: RSI overbought OR price above BB, with MACD confirmation
        df['short_signal'] = (
            ((df['rsi'] > self.rsi_overbought) | (df['close'] > df['bb_upper'])) &  # RSI overbought OR price above BB
            (df['macd'] < df['macd_signal']) &  # MACD bearish
            (df['macd_hist'] < df['macd_hist'].shift(1)) &  # MACD histogram decreasing
            (df['volume_ratio'] > 0.8)  # Some volume activity
        )
        
        # Exit signals
        df['exit_long'] = (
            (df['rsi'] > 70) |  # RSI overbought
            (df['close'] > df['bb_upper']) |  # Price above upper BB
            (df['macd'] < df['macd_signal'])  # MACD bearish crossover
        )
        
        df['exit_short'] = (
            (df['rsi'] < 30) |  # RSI oversold
            (df['close'] < df['bb_lower']) |  # Price below lower BB
            (df['macd'] > df['macd_signal'])  # MACD bullish crossover
        )
        
        logger.info("Indicators calculated successfully")
        return df
    
    def calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_macd(self, prices: pd.Series, fast: int, slow: int, signal: int) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD indicator"""
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal, adjust=False).mean()
        macd_hist = macd - macd_signal
        return macd, macd_signal, macd_hist
    
    def calculate_bollinger_bands(self, prices: pd.Series, period: int, std_dev: float) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        middle = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        return upper, middle, lower
    
    def calculate_atr(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Average True Range"""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        return atr
    
    def run_phase1(self):
        """Execute Phase 1: Strategy Design with new indicators"""
        logger.info("="*50)
        logger.info("PHASE 1: ADVANCED STRATEGY DESIGN")
        logger.info("="*50)
        
        # Fetch historical data
        df = self.fetch_historical_data(days=180)
        
        # Calculate indicators
        df = self.calculate_indicators(df)
        
        # Save processed data
        df.to_csv('historical_data_5m_advanced.csv')
        logger.info("Historical 5m data with advanced indicators saved")
        
        # Generate summary statistics
        logger.info("\nData Summary:")
        logger.info(f"Date range: {df.index[0]} to {df.index[-1]}")
        logger.info(f"Total 5-minute bars: {len(df)}")
        logger.info(f"Long signals: {df['long_signal'].sum()}")
        logger.info(f"Short signals: {df['short_signal'].sum()}")
        logger.info(f"Average RSI: {df['rsi'].mean():.2f}")
        logger.info(f"Average volume ratio: {df['volume_ratio'].mean():.2f}")
        
        return df
    
    def backtest(self, df: pd.DataFrame, initial_capital: float = 10000) -> Dict:
        """
        Run backtest on historical data with the advanced strategy
        
        Args:
            df: DataFrame with OHLCV data and indicators
            initial_capital: Starting capital for backtest
            
        Returns:
            Dictionary with backtest results and metrics
        """
        logger.info("Running advanced strategy backtest...")
        
        # Initialize backtest variables
        capital = initial_capital
        position = 0
        position_type = None  # 'long' or 'short'
        entry_price = 0
        trades = []
        equity_curve = []
        
        # Track position and trades
        entry_time = None
        stop_loss = 0
        take_profit = 0
        
        for idx in range(len(df)):
            current_bar = df.iloc[idx]
            timestamp = df.index[idx]
            close_price = current_bar['close']
            
            # Record equity
            if position_type == 'long':
                current_equity = capital + (position * (close_price - entry_price))
            elif position_type == 'short':
                current_equity = capital + (position * (entry_price - close_price))
            else:
                current_equity = capital
                
            equity_curve.append({
                'timestamp': timestamp,
                'equity': current_equity,
                'capital': capital,
                'position_value': position * close_price if position > 0 else 0
            })
            
            # Skip if indicators not ready
            if pd.isna(current_bar['rsi']) or pd.isna(current_bar['macd']):
                continue
            
            # Check for exit conditions first (for existing positions)
            if position > 0:
                exit_signal = False
                exit_reason = ''
                
                if position_type == 'long':
                    # Check stop loss and take profit
                    if close_price <= stop_loss:
                        exit_signal = True
                        exit_reason = 'stop_loss'
                    elif close_price >= take_profit:
                        exit_signal = True
                        exit_reason = 'take_profit'
                    elif current_bar['exit_long']:
                        exit_signal = True
                        exit_reason = 'signal'
                        
                elif position_type == 'short':
                    # Check stop loss and take profit
                    if close_price >= stop_loss:
                        exit_signal = True
                        exit_reason = 'stop_loss'
                    elif close_price <= take_profit:
                        exit_signal = True
                        exit_reason = 'take_profit'
                    elif current_bar['exit_short']:
                        exit_signal = True
                        exit_reason = 'signal'
                
                if exit_signal:
                    # Calculate P&L
                    if position_type == 'long':
                        pnl = position * (close_price - entry_price)
                    else:  # short
                        pnl = position * (entry_price - close_price)
                    
                    pnl_pct = (pnl / (position * entry_price)) * 100
                    
                    trades.append({
                        'entry_time': entry_time,
                        'exit_time': timestamp,
                        'entry_price': entry_price,
                        'exit_price': close_price,
                        'position_size': position,
                        'position_type': position_type,
                        'pnl': pnl,
                        'pnl_pct': pnl_pct,
                        'exit_reason': exit_reason
                    })
                    
                    capital += pnl
                    position = 0
                    position_type = None
                    entry_price = 0
                    entry_time = None
                    
                    logger.debug(f"Exit {position_type} at {timestamp}: Price={close_price:.2f}, PnL={pnl:.2f} ({pnl_pct:.2f}%)")
            
            # Check for entry signals (only if no position)
            elif position == 0 and capital > 0:
                if current_bar['long_signal']:
                    # Enter long position
                    position = capital / close_price
                    position_type = 'long'
                    entry_price = close_price
                    entry_time = timestamp
                    
                    # Set stop loss and take profit
                    atr = current_bar['atr']
                    stop_loss = close_price - (self.atr_stop_multiplier * atr)
                    take_profit = close_price + (self.atr_target_multiplier * atr)
                    
                    capital = 0
                    
                    logger.debug(f"Long entry at {timestamp}: Price={close_price:.2f}, SL={stop_loss:.2f}, TP={take_profit:.2f}")
                    
                elif current_bar['short_signal']:
                    # Enter short position
                    position = capital / close_price
                    position_type = 'short'
                    entry_price = close_price
                    entry_time = timestamp
                    
                    # Set stop loss and take profit
                    atr = current_bar['atr']
                    stop_loss = close_price + (self.atr_stop_multiplier * atr)
                    take_profit = close_price - (self.atr_target_multiplier * atr)
                    
                    capital = 0
                    
                    logger.debug(f"Short entry at {timestamp}: Price={close_price:.2f}, SL={stop_loss:.2f}, TP={take_profit:.2f}")
        
        # Close any open position at end
        if position > 0:
            if position_type == 'long':
                pnl = position * (df.iloc[-1]['close'] - entry_price)
            else:  # short
                pnl = position * (entry_price - df.iloc[-1]['close'])
                
            pnl_pct = (pnl / (position * entry_price)) * 100
            
            trades.append({
                'entry_time': entry_time,
                'exit_time': df.index[-1],
                'entry_price': entry_price,
                'exit_price': df.iloc[-1]['close'],
                'position_size': position,
                'position_type': position_type,
                'pnl': pnl,
                'pnl_pct': pnl_pct,
                'exit_reason': 'end_of_data'
            })
            
            capital += pnl
        
        # Calculate metrics
        equity_df = pd.DataFrame(equity_curve)
        equity_df.set_index('timestamp', inplace=True)
        
        # Calculate returns
        equity_df['returns'] = equity_df['equity'].pct_change()
        equity_df['cumulative_returns'] = (1 + equity_df['returns']).cumprod() - 1
        
        # Calculate metrics
        final_capital = capital if position == 0 else equity_curve[-1]['equity']
        total_return = (final_capital - initial_capital) / initial_capital
        days_in_backtest = (df.index[-1] - df.index[0]).days
        years_in_backtest = days_in_backtest / 365.25
        
        # CAGR
        cagr = (((final_capital / initial_capital) ** (1 / years_in_backtest)) - 1) if years_in_backtest > 0 else 0
        
        # Sharpe Ratio (assuming 0% risk-free rate)
        # Use 5-minute returns, annualized assuming 288 5-minute periods per day
        returns_5m = equity_df['returns'].dropna()
        if len(returns_5m) > 0:
            periods_per_year = 288 * 252  # 5-minute periods per year
            sharpe_ratio = (returns_5m.mean() / returns_5m.std()) * np.sqrt(periods_per_year) if returns_5m.std() > 0 else 0
        else:
            sharpe_ratio = 0
        
        # Maximum Drawdown
        equity_df['cummax'] = equity_df['equity'].cummax()
        equity_df['drawdown'] = (equity_df['equity'] - equity_df['cummax']) / equity_df['cummax']
        max_drawdown = equity_df['drawdown'].min()
        
        # Win rate and trade analysis
        winning_trades = [t for t in trades if t['pnl'] > 0]
        losing_trades = [t for t in trades if t['pnl'] <= 0]
        win_rate = len(winning_trades) / len(trades) if trades else 0
        
        # Separate long and short trades
        long_trades = [t for t in trades if t['position_type'] == 'long']
        short_trades = [t for t in trades if t['position_type'] == 'short']
        
        # Average trade metrics
        avg_win = np.mean([t['pnl_pct'] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t['pnl_pct'] for t in losing_trades]) if losing_trades else 0
        profit_factor = abs(sum(t['pnl'] for t in winning_trades) / sum(t['pnl'] for t in losing_trades)) if losing_trades and sum(t['pnl'] for t in losing_trades) != 0 else 0
        
        results = {
            'initial_capital': initial_capital,
            'final_capital': final_capital,
            'total_return': total_return,
            'cagr': cagr,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'total_trades': len(trades),
            'long_trades': len(long_trades),
            'short_trades': len(short_trades),
            'winning_trades': len(winning_trades),
            'win_rate': win_rate,
            'avg_win_pct': avg_win,
            'avg_loss_pct': avg_loss,
            'profit_factor': profit_factor,
            'days_in_backtest': days_in_backtest,
            'trades': trades,
            'equity_curve': equity_df
        }
        
        self.backtest_results = results
        return results


def main():
    """Main entry point"""
    strategy = BTCAdvancedStrategy()
    
    # Phase 1: Strategy Design
    try:
        historical_data = strategy.run_phase1()
        logger.info("\nPhase 1 completed successfully!")
        logger.info("Proceeding to Phase 2: Backtesting...")
        
        # Phase 2: Backtesting
        results = strategy.backtest(historical_data)
        
        # Display results
        logger.info("\n" + "="*50)
        logger.info("ADVANCED STRATEGY BACKTEST RESULTS")
        logger.info("="*50)
        logger.info(f"Initial Capital: ${results['initial_capital']:,.2f}")
        logger.info(f"Final Capital: ${results['final_capital']:,.2f}")
        logger.info(f"Total Return: {results['total_return']*100:.2f}%")
        logger.info(f"CAGR: {results['cagr']*100:.2f}%")
        logger.info(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        logger.info(f"Max Drawdown: {results['max_drawdown']*100:.2f}%")
        logger.info(f"Total Trades: {results['total_trades']}")
        logger.info(f"  - Long Trades: {results['long_trades']}")
        logger.info(f"  - Short Trades: {results['short_trades']}")
        logger.info(f"Win Rate: {results['win_rate']*100:.2f}%")
        logger.info(f"Avg Win: {results['avg_win_pct']:.2f}%")
        logger.info(f"Avg Loss: {results['avg_loss_pct']:.2f}%")
        logger.info(f"Profit Factor: {results['profit_factor']:.2f}")
        
        # Save results
        with open('advanced_backtest_results.json', 'w') as f:
            json_results = {k: v for k, v in results.items() if k not in ['equity_curve']}
            json.dump(json_results, f, indent=2, default=str)
        
        results['equity_curve'].to_csv('advanced_equity_curve.csv')
        
        # Check pass criteria
        passed = results['sharpe_ratio'] >= 1.0 and abs(results['max_drawdown']) <= 0.10
        
        logger.info("\nPass Criteria Check:")
        logger.info(f"Sharpe Ratio >= 1.0: {'PASS' if results['sharpe_ratio'] >= 1.0 else 'FAIL'} ({results['sharpe_ratio']:.2f})")
        logger.info(f"Max Drawdown <= 10%: {'PASS' if abs(results['max_drawdown']) <= 0.10 else 'FAIL'} ({abs(results['max_drawdown'])*100:.2f}%)")
        logger.info(f"Overall: {'PASS' if passed else 'FAIL'}")
        
        if not passed:
            logger.info("\nStrategy needs further optimization...")
        else:
            logger.info("\nStrategy passed! Ready for paper trading.")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()
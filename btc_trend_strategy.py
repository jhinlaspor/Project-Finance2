#!/usr/bin/env python3
"""
BTC Trend Following Strategy for Alpaca

This script implements a trend-following strategy using:
- EMA crossovers for trend direction
- RSI for momentum confirmation
- ATR for volatility-based position sizing and stops
- Volume confirmation

Uses 5-minute timeframe with strict risk management.

Author: Claude Opus 4 Agent
Date: January 2025
"""

import os
import sys
import time
import json
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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trend_trading_system.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class BTCTrendStrategy:
    """Trend following strategy with proper risk management"""
    
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
        
        # Trend indicators
        self.ema_fast = 20
        self.ema_slow = 50
        self.ema_trend = 200  # Long-term trend filter
        
        # Momentum
        self.rsi_period = 14
        self.rsi_threshold = 45  # Only trade when RSI > 45 (bullish momentum)
        
        # Volatility
        self.atr_period = 14
        self.atr_stop_multiplier = 2.0
        self.position_size_atr_factor = 0.02  # Risk 2% per trade
        
        # Volume
        self.volume_ma_period = 50
        self.volume_threshold = 1.0  # Volume must be above average
        
        # Risk management
        self.max_position_pct = 0.95  # Max 95% of capital in a position
        self.max_daily_loss_pct = 0.05  # Max 5% daily loss
        
        logger.info(f"Trend strategy initialized")
    
    def fetch_and_prepare_data(self, days: int = 180) -> pd.DataFrame:
        """Fetch 1-minute data and aggregate to 5-minute bars"""
        logger.info(f"Fetching {days} days of data for {self.symbol}")
        
        # Fetch data in chunks
        chunk_days = 30
        all_data = []
        end_time = datetime(2025, 1, 15, 0, 0, 0)
        
        for i in range(0, days, chunk_days):
            chunk_start = end_time - timedelta(days=min(i + chunk_days, days))
            chunk_end = end_time - timedelta(days=i)
            
            logger.info(f"Fetching chunk: {chunk_start.strftime('%Y-%m-%d')} to {chunk_end.strftime('%Y-%m-%d')}")
            
            request_params = CryptoBarsRequest(
                symbol_or_symbols=[self.symbol],
                timeframe=TimeFrame.Minute,
                start=chunk_start,
                end=chunk_end
            )
            
            try:
                bars = self.crypto_client.get_crypto_bars(request_params)
                if bars.df is not None and not bars.df.empty:
                    all_data.append(bars.df)
                    logger.info(f"Fetched {len(bars.df)} bars")
                time.sleep(0.5)
            except Exception as e:
                logger.error(f"Error fetching chunk: {e}")
                continue
        
        if not all_data:
            raise ValueError("No data fetched")
        
        # Combine and clean data
        df = pd.concat(all_data)
        df = df[~df.index.duplicated(keep='first')]
        df = df.sort_index()
        df = df[['open', 'high', 'low', 'close', 'volume']]
        
        # Handle multi-index
        if isinstance(df.index, pd.MultiIndex):
            timestamps = [idx[1] if isinstance(idx, tuple) else idx for idx in df.index]
            df.index = pd.DatetimeIndex(timestamps)
        
        # Aggregate to 5-minute
        df_5m = df.resample('5min').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        
        logger.info(f"Aggregated to {len(df_5m)} 5-minute bars")
        return df_5m
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators"""
        logger.info("Calculating indicators...")
        
        # EMAs
        df['ema_fast'] = df['close'].ewm(span=self.ema_fast, adjust=False).mean()
        df['ema_slow'] = df['close'].ewm(span=self.ema_slow, adjust=False).mean()
        df['ema_trend'] = df['close'].ewm(span=self.ema_trend, adjust=False).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # ATR
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = true_range.rolling(window=self.atr_period).mean()
        
        # Volume
        df['volume_ma'] = df['volume'].rolling(window=self.volume_ma_period).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        # Trend strength
        df['trend_strength'] = (df['ema_fast'] - df['ema_slow']) / df['atr']
        
        # Generate signals
        df['bullish_trend'] = (
            (df['ema_fast'] > df['ema_slow']) &  # Fast EMA above slow
            (df['ema_slow'] > df['ema_trend']) &  # Slow EMA above trend
            (df['close'] > df['ema_trend'])  # Price above long-term trend
        )
        
        df['long_entry'] = (
            (df['bullish_trend']) &  # In bullish trend
            (df['bullish_trend'].shift(1) == False) &  # Just turned bullish
            (df['rsi'] > self.rsi_threshold) &  # Momentum confirmation
            (df['volume_ratio'] >= self.volume_threshold)  # Volume confirmation
        )
        
        df['long_exit'] = (
            (df['ema_fast'] < df['ema_slow']) |  # Trend reversal
            (df['close'] < df['ema_trend'])  # Price below long-term trend
        )
        
        logger.info("Indicators calculated")
        return df
    
    def backtest(self, df: pd.DataFrame, initial_capital: float = 10000) -> Dict:
        """Run backtest with proper risk management"""
        logger.info("Running backtest...")
        
        # Initialize variables
        capital = initial_capital
        position = 0
        entry_price = 0
        stop_loss = 0
        trades = []
        equity_curve = []
        daily_pnl = []
        
        # Track daily performance
        current_date = None
        daily_start_capital = capital
        
        for idx in range(len(df)):
            current_bar = df.iloc[idx]
            timestamp = df.index[idx]
            close_price = current_bar['close']
            
            # Check for new day
            if current_date != timestamp.date():
                if current_date is not None:
                    # Calculate daily P&L
                    daily_return = (capital - daily_start_capital) / daily_start_capital
                    daily_pnl.append({
                        'date': current_date,
                        'return': daily_return,
                        'capital': capital
                    })
                    
                    # Check daily loss limit
                    if daily_return <= -self.max_daily_loss_pct:
                        logger.warning(f"Daily loss limit hit on {current_date}")
                        # Skip trading for the rest of the day
                        continue
                
                current_date = timestamp.date()
                daily_start_capital = capital
            
            # Calculate current equity
            if position > 0:
                current_equity = capital + (position * (close_price - entry_price))
            else:
                current_equity = capital
            
            equity_curve.append({
                'timestamp': timestamp,
                'equity': current_equity,
                'capital': capital,
                'position': position
            })
            
            # Skip if indicators not ready
            if pd.isna(current_bar['atr']) or pd.isna(current_bar['rsi']):
                continue
            
            # Exit logic
            if position > 0:
                # Check stop loss
                if close_price <= stop_loss or current_bar['long_exit']:
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
                        'exit_reason': 'stop_loss' if close_price <= stop_loss else 'signal'
                    })
                    
                    capital = exit_value
                    position = 0
                    entry_price = 0
                    stop_loss = 0
            
            # Entry logic
            elif current_bar['long_entry'] and capital > 100:  # Min capital check
                # Calculate position size based on ATR
                atr = current_bar['atr']
                risk_amount = capital * self.position_size_atr_factor
                position_size_by_risk = risk_amount / (self.atr_stop_multiplier * atr)
                
                # Limit position size
                max_position_value = capital * self.max_position_pct
                max_position_size = max_position_value / close_price
                
                position = min(position_size_by_risk, max_position_size)
                entry_price = close_price
                entry_time = timestamp
                stop_loss = close_price - (self.atr_stop_multiplier * atr)
                
                # Deduct capital
                capital = capital - (position * entry_price)
                
                logger.debug(f"Entry at {timestamp}: Price={close_price:.2f}, Position={position:.6f}, Stop={stop_loss:.2f}")
        
        # Close any open position
        if position > 0:
            exit_value = position * df.iloc[-1]['close']
            pnl = exit_value - (position * entry_price)
            pnl_pct = (pnl / (position * entry_price)) * 100
            
            trades.append({
                'entry_time': entry_time,
                'exit_time': df.index[-1],
                'entry_price': entry_price,
                'exit_price': df.iloc[-1]['close'],
                'position_size': position,
                'pnl': pnl,
                'pnl_pct': pnl_pct,
                'exit_reason': 'end_of_data'
            })
            
            capital = capital + exit_value
        
        # Calculate metrics
        equity_df = pd.DataFrame(equity_curve)
        equity_df.set_index('timestamp', inplace=True)
        
        # Final metrics
        final_capital = capital
        total_return = (final_capital - initial_capital) / initial_capital
        
        # Calculate other metrics
        if len(trades) > 0:
            winning_trades = [t for t in trades if t['pnl'] > 0]
            losing_trades = [t for t in trades if t['pnl'] <= 0]
            win_rate = len(winning_trades) / len(trades)
            avg_win = np.mean([t['pnl_pct'] for t in winning_trades]) if winning_trades else 0
            avg_loss = np.mean([t['pnl_pct'] for t in losing_trades]) if losing_trades else 0
            
            # Profit factor
            gross_profit = sum(t['pnl'] for t in winning_trades) if winning_trades else 0
            gross_loss = abs(sum(t['pnl'] for t in losing_trades)) if losing_trades else 1
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        else:
            win_rate = avg_win = avg_loss = profit_factor = 0
        
        # Calculate Sharpe ratio
        equity_df['returns'] = equity_df['equity'].pct_change()
        returns = equity_df['returns'].dropna()
        if len(returns) > 0 and returns.std() > 0:
            # Annualize for 5-minute bars
            periods_per_year = 288 * 252
            sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(periods_per_year)
        else:
            sharpe_ratio = 0
        
        # Maximum drawdown
        equity_df['cummax'] = equity_df['equity'].cummax()
        equity_df['drawdown'] = (equity_df['equity'] - equity_df['cummax']) / equity_df['cummax']
        max_drawdown = equity_df['drawdown'].min()
        
        # CAGR
        days = (df.index[-1] - df.index[0]).days
        years = days / 365.25
        cagr = (((final_capital / initial_capital) ** (1 / years)) - 1) if years > 0 and final_capital > 0 else 0
        
        results = {
            'initial_capital': initial_capital,
            'final_capital': final_capital,
            'total_return': total_return,
            'cagr': cagr,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'total_trades': len(trades),
            'winning_trades': len(winning_trades) if len(trades) > 0 else 0,
            'win_rate': win_rate,
            'avg_win_pct': avg_win,
            'avg_loss_pct': avg_loss,
            'profit_factor': profit_factor,
            'trades': trades,
            'equity_curve': equity_df
        }
        
        return results
    
    def save_results(self, results: Dict, prefix: str = "trend"):
        """Save backtest results"""
        # Save metrics
        metrics = {k: v for k, v in results.items() if k not in ['trades', 'equity_curve']}
        with open(f'{prefix}_backtest_results.json', 'w') as f:
            json.dump(metrics, f, indent=2, default=str)
        
        # Save trades
        if results['trades']:
            pd.DataFrame(results['trades']).to_csv(f'{prefix}_trades.csv', index=False)
        
        # Save equity curve
        results['equity_curve'].to_csv(f'{prefix}_equity_curve.csv')
        
        logger.info(f"Results saved with prefix '{prefix}'")


def main():
    """Main entry point"""
    strategy = BTCTrendStrategy()
    
    try:
        # Fetch and prepare data
        df = strategy.fetch_and_prepare_data(days=180)
        
        # Calculate indicators
        df = strategy.calculate_indicators(df)
        
        # Save processed data
        df.to_csv('trend_strategy_data_5m.csv')
        
        # Display summary
        logger.info("\n" + "="*50)
        logger.info("DATA SUMMARY")
        logger.info("="*50)
        logger.info(f"Date range: {df.index[0]} to {df.index[-1]}")
        logger.info(f"Total 5-minute bars: {len(df)}")
        logger.info(f"Long entry signals: {df['long_entry'].sum()}")
        logger.info(f"Average RSI: {df['rsi'].mean():.2f}")
        
        # Run backtest
        results = strategy.backtest(df)
        
        # Display results
        logger.info("\n" + "="*50)
        logger.info("BACKTEST RESULTS")
        logger.info("="*50)
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
        logger.info(f"Profit Factor: {results['profit_factor']:.2f}")
        
        # Save results
        strategy.save_results(results)
        
        # Check pass criteria
        passed = results['sharpe_ratio'] >= 1.0 and abs(results['max_drawdown']) <= 0.10
        
        logger.info("\n" + "="*50)
        logger.info("PASS CRITERIA CHECK")
        logger.info("="*50)
        logger.info(f"Sharpe Ratio >= 1.0: {'PASS' if results['sharpe_ratio'] >= 1.0 else 'FAIL'} ({results['sharpe_ratio']:.2f})")
        logger.info(f"Max Drawdown <= 10%: {'PASS' if abs(results['max_drawdown']) <= 0.10 else 'FAIL'} ({abs(results['max_drawdown'])*100:.2f}%)")
        logger.info(f"Overall: {'PASS' if passed else 'FAIL'}")
        
        if passed:
            logger.info("\nStrategy PASSED! Ready for paper trading.")
        else:
            logger.info("\nStrategy needs optimization.")
            
    except Exception as e:
        logger.error(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()
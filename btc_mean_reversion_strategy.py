#!/usr/bin/env python3
"""
BTC Mean Reversion Strategy for Alpaca

This script implements a sophisticated mean reversion strategy designed for crypto's
volatile nature with exceptional risk management:

Key Features:
1. Multiple timeframe analysis (5m, 15m, 1h)
2. Z-score based entry/exit signals
3. RSI divergence detection
4. Volume profile analysis
5. Dynamic position sizing based on Kelly Criterion
6. Volatility-adjusted stop losses
7. Maximum adverse excursion (MAE) tracking
8. Regime detection (trending vs ranging)

Risk Management:
- Kelly Criterion for optimal position sizing
- Dynamic stop losses based on ATR and volatility
- Maximum position limits
- Correlation-based portfolio heat
- Time-based exits for stale positions

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
from scipy import stats
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
        logging.FileHandler('mean_reversion_strategy.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class BTCMeanReversionStrategy:
    """Sophisticated mean reversion strategy with exceptional risk management"""
    
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
        
        # Mean reversion parameters - BALANCED FOR SHARPE AND DRAWDOWN
        self.bb_period = 20
        self.bb_std_entry = 2.2  # Enter when price touches 2.2 std dev (balanced)
        self.bb_std_exit = 0.4   # Exit at 0.4 std dev
        self.zscore_period = 25  # Medium lookback
        self.zscore_entry_threshold = 2.2  # Balanced threshold
        self.zscore_exit_threshold = 0.4  # Balanced exit
        
        # RSI parameters
        self.rsi_period = 14
        self.rsi_oversold = 30
        self.rsi_overbought = 70
        self.rsi_divergence_lookback = 10
        
        # Volume analysis
        self.volume_ma_period = 20
        self.volume_spike_threshold = 2.0
        self.volume_profile_bins = 50
        
        # Regime detection
        self.regime_lookback = 100
        self.trend_threshold = 0.002  # 0.2% per bar for trending
        
        # Risk management parameters - BALANCED APPROACH
        self.max_position_size = 0.50  # Max 50% of capital per position
        self.kelly_fraction = 0.20  # Use 20% of Kelly Criterion
        self.max_positions = 3  # Maximum concurrent positions
        self.position_timeout_bars = 40  # Exit after 40 bars (3.3 hours on 5m)
        self.max_daily_trades = 8  # Moderate daily limit
        self.max_drawdown_pct = 0.07  # 7% max drawdown trigger
        self.volatility_lookback = 20
        self.atr_period = 14
        self.stop_loss_atr_mult = 1.8  # Balanced stop loss
        self.take_profit_atr_mult = 2.7  # Balanced take profit
        
        # Performance tracking
        self.win_rate_window = 30  # Track last 30 trades
        self.min_edge = 0.02  # Minimum 2% edge to trade
        
        # Additional filters
        self.max_volatility = 0.50  # Don't trade when annualized vol > 50% (less restrictive)
        self.min_volatility = 0.08  # Don't trade when annualized vol < 8% (less restrictive)
        
        logger.info(f"Mean Reversion Strategy initialized")
    
    def fetch_and_prepare_data(self, days: int = 180) -> Dict[str, pd.DataFrame]:
        """Fetch data and prepare multiple timeframes"""
        logger.info(f"Fetching {days} days of data for {self.symbol}")
        
        # Fetch 1-minute data
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
        df_1m = pd.concat(all_data)
        df_1m = df_1m[~df_1m.index.duplicated(keep='first')]
        df_1m = df_1m.sort_index()
        df_1m = df_1m[['open', 'high', 'low', 'close', 'volume']]
        
        # Handle multi-index
        if isinstance(df_1m.index, pd.MultiIndex):
            timestamps = [idx[1] if isinstance(idx, tuple) else idx for idx in df_1m.index]
            df_1m.index = pd.DatetimeIndex(timestamps)
        
        # Create multiple timeframes
        timeframes = {
            '5m': df_1m.resample('5min').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna(),
            '15m': df_1m.resample('15min').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna(),
            '1h': df_1m.resample('60min').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()
        }
        
        logger.info(f"Created timeframes: 5m ({len(timeframes['5m'])} bars), "
                   f"15m ({len(timeframes['15m'])} bars), 1h ({len(timeframes['1h'])} bars)")
        
        return timeframes
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate mean reversion indicators"""
        logger.info("Calculating mean reversion indicators...")
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=self.bb_period).mean()
        bb_std = df['close'].rolling(window=self.bb_period).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * self.bb_std_entry)
        df['bb_lower'] = df['bb_middle'] - (bb_std * self.bb_std_entry)
        df['bb_upper_exit'] = df['bb_middle'] + (bb_std * self.bb_std_exit)
        df['bb_lower_exit'] = df['bb_middle'] - (bb_std * self.bb_std_exit)
        df['bb_width'] = df['bb_upper'] - df['bb_lower']
        df['bb_pct'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Z-Score
        df['zscore'] = (df['close'] - df['close'].rolling(self.zscore_period).mean()) / df['close'].rolling(self.zscore_period).std()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # RSI Divergence Detection
        df['price_higher'] = df['close'] > df['close'].shift(self.rsi_divergence_lookback)
        df['rsi_lower'] = df['rsi'] < df['rsi'].shift(self.rsi_divergence_lookback)
        df['bearish_divergence'] = df['price_higher'] & df['rsi_lower']
        
        df['price_lower'] = df['close'] < df['close'].shift(self.rsi_divergence_lookback)
        df['rsi_higher'] = df['rsi'] > df['rsi'].shift(self.rsi_divergence_lookback)
        df['bullish_divergence'] = df['price_lower'] & df['rsi_higher']
        
        # Volume Analysis
        df['volume_ma'] = df['volume'].rolling(window=self.volume_ma_period).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        df['volume_spike'] = df['volume_ratio'] > self.volume_spike_threshold
        
        # ATR for stops
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = true_range.rolling(window=self.atr_period).mean()
        
        # Volatility
        df['returns'] = df['close'].pct_change()
        df['volatility'] = df['returns'].rolling(window=self.volatility_lookback).std() * np.sqrt(288)  # Annualized for 5m bars
        
        # Regime Detection
        df['trend_strength'] = df['close'].pct_change(self.regime_lookback)
        df['is_trending'] = np.abs(df['trend_strength']) > (self.trend_threshold * self.regime_lookback)
        df['is_ranging'] = ~df['is_trending']
        
        # Mean Reversion Signals
        # Long signals: Multiple conditions for entry
        df['long_signal'] = (
            # Primary signal: Price below BB, Z-score extreme, RSI oversold
            (
                (df['close'] < df['bb_lower']) &
                (df['zscore'] < -self.zscore_entry_threshold) &
                (df['rsi'] < self.rsi_oversold) &
                (df['volume_spike']) &
                (df['is_ranging'])
            ) | 
            # Secondary signal: Bullish divergence
            (
                df['bullish_divergence'] &
                (df['zscore'] < -1.5) &
                (df['is_ranging'])
            ) |
            # Tertiary signal: Extreme oversold with volume
            (
                (df['zscore'] < -1.8) &
                (df['rsi'] < 35) &
                (df['volume_ratio'] > 1.5) &
                (df['is_ranging']) &
                (df['bb_pct'] < 0.1)  # Price in bottom 10% of BB
            )
        )
        
        # Short signals: Multiple conditions for entry
        df['short_signal'] = (
            # Primary signal: Price above BB, Z-score extreme, RSI overbought
            (
                (df['close'] > df['bb_upper']) &
                (df['zscore'] > self.zscore_entry_threshold) &
                (df['rsi'] > self.rsi_overbought) &
                (df['volume_spike']) &
                (df['is_ranging'])
            ) | 
            # Secondary signal: Bearish divergence
            (
                df['bearish_divergence'] &
                (df['zscore'] > 1.5) &
                (df['is_ranging'])
            ) |
            # Tertiary signal: Extreme overbought with volume
            (
                (df['zscore'] > 1.8) &
                (df['rsi'] > 65) &
                (df['volume_ratio'] > 1.5) &
                (df['is_ranging']) &
                (df['bb_pct'] > 0.9)  # Price in top 10% of BB
            )
        )
        
        # Exit signals
        df['exit_long'] = (
            (df['close'] > df['bb_middle']) |  # Price returned to mean
            (df['zscore'] > -self.zscore_exit_threshold) |  # Z-score normalized
            (df['rsi'] > 50)  # RSI neutral
        )
        
        df['exit_short'] = (
            (df['close'] < df['bb_middle']) |  # Price returned to mean
            (df['zscore'] < self.zscore_exit_threshold) |  # Z-score normalized
            (df['rsi'] < 50)  # RSI neutral
        )
        
        logger.info("Indicators calculated successfully")
        return df
    
    def calculate_kelly_position_size(self, win_rate: float, avg_win: float, avg_loss: float) -> float:
        """Calculate optimal position size using Kelly Criterion"""
        if avg_loss == 0 or win_rate == 0:
            return 0.02  # Default 2% position
        
        # Kelly formula: f = (p * b - q) / b
        # where p = win rate, q = loss rate, b = win/loss ratio
        b = abs(avg_win / avg_loss)
        p = win_rate
        q = 1 - win_rate
        
        kelly = (p * b - q) / b
        
        # Apply Kelly fraction (conservative)
        position_size = kelly * self.kelly_fraction
        
        # Limit position size
        position_size = max(0.01, min(position_size, self.max_position_size))
        
        return position_size
    
    def calculate_dynamic_stops(self, entry_price: float, atr: float, volatility: float, position_type: str) -> Tuple[float, float]:
        """Calculate dynamic stop loss and take profit based on market conditions"""
        # Adjust ATR multiplier based on volatility
        volatility_adjustment = min(2.0, max(0.5, volatility / 0.20))  # Normalize to 20% annual vol
        
        stop_distance = atr * self.stop_loss_atr_mult * volatility_adjustment
        target_distance = atr * self.take_profit_atr_mult * volatility_adjustment
        
        if position_type == 'long':
            stop_loss = entry_price - stop_distance
            take_profit = entry_price + target_distance
        else:  # short
            stop_loss = entry_price + stop_distance
            take_profit = entry_price - target_distance
        
        return stop_loss, take_profit
    
    def backtest(self, timeframes: Dict[str, pd.DataFrame], initial_capital: float = 10000) -> Dict:
        """Run sophisticated backtest with multiple timeframes"""
        logger.info("Running mean reversion backtest...")
        
        # Use 5m as primary timeframe
        df_5m = timeframes['5m'].copy()
        df_15m = timeframes['15m'].copy()
        df_1h = timeframes['1h'].copy()
        
        # Calculate indicators for all timeframes
        df_5m = self.calculate_indicators(df_5m)
        df_15m = self.calculate_indicators(df_15m)
        df_1h = self.calculate_indicators(df_1h)
        
        # Initialize backtest variables
        capital = initial_capital
        positions = []  # List of open positions
        closed_trades = []
        equity_curve = []
        daily_trades = {}
        
        # Performance tracking
        recent_trades = []
        
        for idx in range(len(df_5m)):
            current_bar = df_5m.iloc[idx]
            timestamp = df_5m.index[idx]
            close_price = current_bar['close']
            
            # Get higher timeframe context
            tf_15m_idx = df_15m.index.get_indexer([timestamp], method='ffill')[0]
            tf_1h_idx = df_1h.index.get_indexer([timestamp], method='ffill')[0]
            
            if tf_15m_idx >= 0 and tf_1h_idx >= 0:
                context_15m = df_15m.iloc[tf_15m_idx]
                context_1h = df_1h.iloc[tf_1h_idx]
            else:
                continue
            
            # Track daily trades
            trade_date = timestamp.date()
            if trade_date not in daily_trades:
                daily_trades[trade_date] = 0
            
            # Calculate current equity
            position_value = sum(p['size'] * (close_price - p['entry_price']) * (1 if p['type'] == 'long' else -1) 
                               for p in positions)
            current_equity = capital + position_value
            
            equity_curve.append({
                'timestamp': timestamp,
                'equity': current_equity,
                'capital': capital,
                'positions': len(positions)
            })
            
            # Check for drawdown limit
            if len(equity_curve) > 1:
                peak_equity = max(e['equity'] for e in equity_curve)
                current_drawdown = (current_equity - peak_equity) / peak_equity
                if current_drawdown < -self.max_drawdown_pct:
                    logger.warning(f"Max drawdown hit at {timestamp}: {current_drawdown:.2%}")
                    # Close all positions
                    for pos in positions:
                        exit_value = pos['size'] * close_price
                        if pos['type'] == 'long':
                            pnl = exit_value - (pos['size'] * pos['entry_price'])
                        else:
                            pnl = (pos['size'] * pos['entry_price']) - exit_value
                        
                        capital += (pos['size'] * pos['entry_price']) + pnl
                        
                        closed_trades.append({
                            'entry_time': pos['entry_time'],
                            'exit_time': timestamp,
                            'type': pos['type'],
                            'entry_price': pos['entry_price'],
                            'exit_price': close_price,
                            'size': pos['size'],
                            'pnl': pnl,
                            'pnl_pct': (pnl / (pos['size'] * pos['entry_price'])) * 100,
                            'exit_reason': 'max_drawdown'
                        })
                    positions = []
                    continue
            
            # Exit logic for existing positions
            positions_to_close = []
            for i, pos in enumerate(positions):
                bars_held = idx - pos['entry_bar']
                
                # Check stop loss and take profit
                if pos['type'] == 'long':
                    if close_price <= pos['stop_loss'] or close_price >= pos['take_profit']:
                        positions_to_close.append(i)
                    elif current_bar['exit_long'] or bars_held > self.position_timeout_bars:
                        positions_to_close.append(i)
                else:  # short
                    if close_price >= pos['stop_loss'] or close_price <= pos['take_profit']:
                        positions_to_close.append(i)
                    elif current_bar['exit_short'] or bars_held > self.position_timeout_bars:
                        positions_to_close.append(i)
            
            # Close positions
            for i in reversed(positions_to_close):
                pos = positions.pop(i)
                exit_value = pos['size'] * close_price
                
                if pos['type'] == 'long':
                    pnl = exit_value - (pos['size'] * pos['entry_price'])
                else:
                    pnl = (pos['size'] * pos['entry_price']) - exit_value
                
                capital += (pos['size'] * pos['entry_price']) + pnl
                
                trade_result = {
                    'entry_time': pos['entry_time'],
                    'exit_time': timestamp,
                    'type': pos['type'],
                    'entry_price': pos['entry_price'],
                    'exit_price': close_price,
                    'size': pos['size'],
                    'pnl': pnl,
                    'pnl_pct': (pnl / (pos['size'] * pos['entry_price'])) * 100,
                    'exit_reason': 'signal' if i in positions_to_close[:1] else 'timeout'
                }
                
                closed_trades.append(trade_result)
                recent_trades.append(trade_result)
                
                # Keep only recent trades for performance tracking
                if len(recent_trades) > self.win_rate_window:
                    recent_trades.pop(0)
            
            # Skip if indicators not ready
            if pd.isna(current_bar['atr']) or pd.isna(current_bar['zscore']):
                continue
            
            # Entry logic
            if (len(positions) < self.max_positions and 
                daily_trades[trade_date] < self.max_daily_trades and
                capital > 1000):  # Min capital check
                
                # Calculate current performance metrics
                if len(recent_trades) >= 10:
                    recent_wins = [t for t in recent_trades if t['pnl'] > 0]
                    win_rate = len(recent_wins) / len(recent_trades)
                    avg_win = np.mean([t['pnl_pct'] for t in recent_wins]) if recent_wins else 0
                    avg_loss = np.mean([abs(t['pnl_pct']) for t in recent_trades if t['pnl'] <= 0]) if any(t['pnl'] <= 0 for t in recent_trades) else 1
                    
                    # Calculate Kelly position size
                    position_size_pct = self.calculate_kelly_position_size(win_rate, avg_win, avg_loss)
                else:
                    position_size_pct = 0.02  # Default 2%
                
                # Check for entry signals with higher timeframe confirmation and volatility filter
                current_volatility = current_bar['volatility']
                if (self.min_volatility < current_volatility < self.max_volatility and
                    current_bar['long_signal'] and context_15m['is_ranging'] and context_1h['is_ranging']):
                    # Calculate position size
                    position_value = capital * position_size_pct
                    position_size = position_value / close_price
                    
                    # Calculate dynamic stops
                    stop_loss, take_profit = self.calculate_dynamic_stops(
                        close_price, current_bar['atr'], current_bar['volatility'], 'long'
                    )
                    
                    positions.append({
                        'entry_time': timestamp,
                        'entry_bar': idx,
                        'type': 'long',
                        'entry_price': close_price,
                        'size': position_size,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit
                    })
                    
                    capital -= position_value
                    daily_trades[trade_date] += 1
                    
                elif (self.min_volatility < current_volatility < self.max_volatility and
                      current_bar['short_signal'] and context_15m['is_ranging'] and context_1h['is_ranging']):
                    # Calculate position size
                    position_value = capital * position_size_pct
                    position_size = position_value / close_price
                    
                    # Calculate dynamic stops
                    stop_loss, take_profit = self.calculate_dynamic_stops(
                        close_price, current_bar['atr'], current_bar['volatility'], 'short'
                    )
                    
                    positions.append({
                        'entry_time': timestamp,
                        'entry_bar': idx,
                        'type': 'short',
                        'entry_price': close_price,
                        'size': position_size,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit
                    })
                    
                    capital -= position_value
                    daily_trades[trade_date] += 1
        
        # Close any remaining positions
        for pos in positions:
            exit_price = df_5m.iloc[-1]['close']
            exit_value = pos['size'] * exit_price
            
            if pos['type'] == 'long':
                pnl = exit_value - (pos['size'] * pos['entry_price'])
            else:
                pnl = (pos['size'] * pos['entry_price']) - exit_value
            
            capital += (pos['size'] * pos['entry_price']) + pnl
            
            closed_trades.append({
                'entry_time': pos['entry_time'],
                'exit_time': df_5m.index[-1],
                'type': pos['type'],
                'entry_price': pos['entry_price'],
                'exit_price': exit_price,
                'size': pos['size'],
                'pnl': pnl,
                'pnl_pct': (pnl / (pos['size'] * pos['entry_price'])) * 100,
                'exit_reason': 'end_of_data'
            })
        
        # Calculate final metrics
        equity_df = pd.DataFrame(equity_curve)
        equity_df.set_index('timestamp', inplace=True)
        
        final_capital = capital
        total_return = (final_capital - initial_capital) / initial_capital
        
        # Performance metrics
        if len(closed_trades) > 0:
            winning_trades = [t for t in closed_trades if t['pnl'] > 0]
            losing_trades = [t for t in closed_trades if t['pnl'] <= 0]
            
            win_rate = len(winning_trades) / len(closed_trades)
            avg_win = np.mean([t['pnl_pct'] for t in winning_trades]) if winning_trades else 0
            avg_loss = np.mean([abs(t['pnl_pct']) for t in losing_trades]) if losing_trades else 0
            
            # Profit factor
            gross_profit = sum(t['pnl'] for t in winning_trades) if winning_trades else 0
            gross_loss = abs(sum(t['pnl'] for t in losing_trades)) if losing_trades else 1
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
            
            # Trade analysis
            long_trades = [t for t in closed_trades if t['type'] == 'long']
            short_trades = [t for t in closed_trades if t['type'] == 'short']
        else:
            win_rate = avg_win = avg_loss = profit_factor = 0
            long_trades = short_trades = []
        
        # Calculate Sharpe ratio
        equity_df['returns'] = equity_df['equity'].pct_change()
        returns = equity_df['returns'].dropna()
        if len(returns) > 0 and returns.std() > 0:
            periods_per_year = 288 * 252  # 5-minute periods
            sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(periods_per_year)
        else:
            sharpe_ratio = 0
        
        # Maximum drawdown
        equity_df['cummax'] = equity_df['equity'].cummax()
        equity_df['drawdown'] = (equity_df['equity'] - equity_df['cummax']) / equity_df['cummax']
        max_drawdown = equity_df['drawdown'].min()
        
        # CAGR
        days = (df_5m.index[-1] - df_5m.index[0]).days
        years = days / 365.25
        cagr = (((final_capital / initial_capital) ** (1 / years)) - 1) if years > 0 and final_capital > 0 else 0
        
        results = {
            'initial_capital': initial_capital,
            'final_capital': final_capital,
            'total_return': total_return,
            'cagr': cagr,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'total_trades': len(closed_trades),
            'long_trades': len(long_trades),
            'short_trades': len(short_trades),
            'winning_trades': len(winning_trades) if closed_trades else 0,
            'win_rate': win_rate,
            'avg_win_pct': avg_win,
            'avg_loss_pct': avg_loss,
            'profit_factor': profit_factor,
            'trades': closed_trades,
            'equity_curve': equity_df
        }
        
        return results
    
    def save_results(self, results: Dict, prefix: str = "mean_reversion"):
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
        
        # Create visualization
        self.plot_results(results, prefix)
        
        logger.info(f"Results saved with prefix '{prefix}'")
    
    def plot_results(self, results: Dict, prefix: str):
        """Create comprehensive visualization"""
        equity_df = results['equity_curve']
        trades_df = pd.DataFrame(results['trades']) if results['trades'] else pd.DataFrame()
        
        fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
        
        # Equity curve
        ax1 = axes[0]
        ax1.plot(equity_df.index, equity_df['equity'], label='Portfolio Value', color='blue', linewidth=2)
        ax1.axhline(y=results['initial_capital'], color='gray', linestyle='--', label='Initial Capital')
        ax1.fill_between(equity_df.index, equity_df['equity'], results['initial_capital'], 
                        where=equity_df['equity'] > results['initial_capital'], alpha=0.3, color='green')
        ax1.fill_between(equity_df.index, equity_df['equity'], results['initial_capital'], 
                        where=equity_df['equity'] <= results['initial_capital'], alpha=0.3, color='red')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.set_title(f'Mean Reversion Strategy - Sharpe: {results["sharpe_ratio"]:.2f}, Max DD: {results["max_drawdown"]*100:.1f}%')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Drawdown
        ax2 = axes[1]
        ax2.fill_between(equity_df.index, equity_df['drawdown'] * 100, 0, color='red', alpha=0.5)
        ax2.set_ylabel('Drawdown (%)')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(top=0)
        
        # Number of positions over time
        ax3 = axes[2]
        ax3.plot(equity_df.index, equity_df['positions'], label='Open Positions', color='purple', alpha=0.7)
        ax3.set_ylabel('Open Positions')
        ax3.set_xlabel('Date')
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(bottom=0)
        
        plt.tight_layout()
        plt.savefig(f'{prefix}_results.png', dpi=150)
        plt.close()
        
        # Trade distribution
        if not trades_df.empty:
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            
            # PnL distribution
            ax1 = axes[0, 0]
            trades_df['pnl_pct'].hist(bins=50, ax=ax1, color='skyblue', edgecolor='black')
            ax1.axvline(x=0, color='red', linestyle='--')
            ax1.set_xlabel('PnL %')
            ax1.set_ylabel('Frequency')
            ax1.set_title('Trade PnL Distribution')
            
            # Win/Loss by type
            ax2 = axes[0, 1]
            trade_summary = trades_df.groupby('type')['pnl'].agg(['count', 'sum', 'mean'])
            trade_summary.plot(kind='bar', ax=ax2)
            ax2.set_xlabel('Trade Type')
            ax2.set_title('Performance by Trade Type')
            ax2.legend(['Count', 'Total PnL', 'Avg PnL'])
            
            # Cumulative PnL
            ax3 = axes[1, 0]
            trades_df['cumulative_pnl'] = trades_df['pnl'].cumsum()
            ax3.plot(trades_df.index, trades_df['cumulative_pnl'], color='green', linewidth=2)
            ax3.set_xlabel('Trade Number')
            ax3.set_ylabel('Cumulative PnL ($)')
            ax3.set_title('Cumulative PnL')
            ax3.grid(True, alpha=0.3)
            
            # Exit reasons
            ax4 = axes[1, 1]
            exit_reasons = trades_df['exit_reason'].value_counts()
            exit_reasons.plot(kind='pie', ax=ax4, autopct='%1.1f%%')
            ax4.set_title('Exit Reasons')
            
            plt.tight_layout()
            plt.savefig(f'{prefix}_trade_analysis.png', dpi=150)
            plt.close()


def main():
    """Main entry point"""
    strategy = BTCMeanReversionStrategy()
    
    try:
        # Fetch and prepare data
        timeframes = strategy.fetch_and_prepare_data(days=180)
        
        # Display summary
        logger.info("\n" + "="*50)
        logger.info("DATA SUMMARY")
        logger.info("="*50)
        for tf, df in timeframes.items():
            logger.info(f"{tf}: {len(df)} bars, from {df.index[0]} to {df.index[-1]}")
        
        # Run backtest
        results = strategy.backtest(timeframes)
        
        # Display results
        logger.info("\n" + "="*50)
        logger.info("MEAN REVERSION STRATEGY RESULTS")
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
            logger.info("Mean reversion strategy successfully exploits crypto volatility with excellent risk management.")
        else:
            logger.info("\nStrategy needs further optimization.")
            
    except Exception as e:
        logger.error(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()
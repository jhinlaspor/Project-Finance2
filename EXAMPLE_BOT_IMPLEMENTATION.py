#!/usr/bin/env python3
"""
Complete Options Trading Bot Example for Alpaca
===============================================

This is a comprehensive example implementation of an options trading bot
using Alpaca's Trading API. This bot implements:

1. Iron Condor strategy for neutral market conditions
2. Real-time risk management
3. Position monitoring and automated exit
4. Comprehensive logging and error handling

DISCLAIMER: This is for educational purposes only. Options trading involves
substantial risk and is not suitable for all investors. Always test
thoroughly in paper trading before using real money.

Author: AI Assistant
Date: 2025
License: MIT
"""

import os
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq

# Alpaca imports
from alpaca.trading.client import TradingClient
from alpaca.data.historical.option import OptionHistoricalDataClient
from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.trading.requests import (
    GetOptionContractsRequest, 
    MarketOrderRequest,
    OptionLegRequest
)
from alpaca.data.requests import (
    OptionLatestQuoteRequest,
    StockLatestTradeRequest
)
from alpaca.trading.enums import (
    AssetStatus, 
    ExerciseStyle, 
    OrderSide, 
    OrderClass, 
    TimeInForce,
    ContractType
)

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('options_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class BotConfig:
    """Configuration for the options trading bot"""
    alpaca_api_key: str
    alpaca_secret_key: str
    paper_trading: bool = True
    max_daily_loss: float = 1000.0
    max_position_size_pct: float = 0.05
    max_portfolio_delta: float = 500.0
    target_dte: int = 30
    min_credit_pct: float = 0.33
    wing_width: float = 5.0
    target_delta: float = 0.20
    check_interval: int = 300  # 5 minutes

@dataclass
class OptionQuote:
    """Option quote data structure"""
    symbol: str
    bid_price: float
    ask_price: float
    mid_price: float
    bid_size: int
    ask_size: int
    timestamp: datetime

@dataclass
class OptionGreeks:
    """Option Greeks data structure"""
    delta: float
    gamma: float
    theta: float
    vega: float
    implied_volatility: float

@dataclass
class IronCondorPosition:
    """Iron Condor position tracking"""
    underlying_symbol: str
    expiration_date: str
    short_call_symbol: str
    long_call_symbol: str
    short_put_symbol: str
    long_put_symbol: str
    net_credit: float
    max_profit: float
    max_loss: float
    break_even_upper: float
    break_even_lower: float
    entry_time: datetime
    quantity: int = 1

class OptionsDataManager:
    """Manages options data retrieval and calculations"""
    
    def __init__(self, config: BotConfig):
        self.config = config
        self.option_client = OptionHistoricalDataClient(
            api_key=config.alpaca_api_key,
            secret_key=config.alpaca_secret_key
        )
        self.stock_client = StockHistoricalDataClient(
            api_key=config.alpaca_api_key,
            secret_key=config.alpaca_secret_key
        )
    
    def get_underlying_price(self, symbol: str) -> float:
        """Get current underlying stock price"""
        try:
            request = StockLatestTradeRequest(symbol_or_symbols=symbol)
            response = self.stock_client.get_stock_latest_trade(request)
            return float(response[symbol].price)
        except Exception as e:
            logger.error(f"Error getting underlying price for {symbol}: {e}")
            raise
    
    async def get_option_quote(self, symbol: str) -> Optional[OptionQuote]:
        """Get latest option quote"""
        try:
            request = OptionLatestQuoteRequest(symbol_or_symbols=[symbol])
            response = self.option_client.get_option_latest_quote(request)
            
            if symbol in response:
                quote_data = response[symbol]
                bid_price = float(quote_data.bid_price)
                ask_price = float(quote_data.ask_price)
                
                return OptionQuote(
                    symbol=symbol,
                    bid_price=bid_price,
                    ask_price=ask_price,
                    mid_price=(bid_price + ask_price) / 2,
                    bid_size=int(quote_data.bid_size),
                    ask_size=int(quote_data.ask_size),
                    timestamp=quote_data.timestamp
                )
            return None
        except Exception as e:
            logger.error(f"Error getting option quote for {symbol}: {e}")
            return None
    
    def calculate_implied_volatility(
        self,
        option_price: float,
        strike_price: float,
        time_to_expiry: float,
        underlying_price: float,
        risk_free_rate: float = 0.045,
        option_type: str = 'call'
    ) -> Optional[float]:
        """Calculate implied volatility using Black-Scholes"""
        
        def black_scholes_price(volatility):
            if volatility <= 0 or time_to_expiry <= 0:
                return float('inf')
                
            d1 = (np.log(underlying_price / strike_price) + 
                  (risk_free_rate + 0.5 * volatility ** 2) * time_to_expiry) / (volatility * np.sqrt(time_to_expiry))
            d2 = d1 - volatility * np.sqrt(time_to_expiry)
            
            if option_type.lower() == 'call':
                price = (underlying_price * norm.cdf(d1) - 
                        strike_price * np.exp(-risk_free_rate * time_to_expiry) * norm.cdf(d2))
            else:
                price = (strike_price * np.exp(-risk_free_rate * time_to_expiry) * norm.cdf(-d2) - 
                        underlying_price * norm.cdf(-d1))
            
            return price - option_price
        
        try:
            # Use Brent's method to find implied volatility
            iv = brentq(black_scholes_price, 0.001, 5.0, maxiter=100, xtol=1e-6)
            return max(iv, 0.001)  # Ensure positive IV
        except (ValueError, RuntimeError):
            logger.warning(f"Could not calculate IV for option price {option_price}")
            return None
    
    def calculate_option_greeks(
        self,
        option_price: float,
        strike_price: float,
        time_to_expiry: float,
        underlying_price: float,
        risk_free_rate: float = 0.045,
        option_type: str = 'call'
    ) -> Optional[OptionGreeks]:
        """Calculate option Greeks"""
        
        try:
            # Calculate implied volatility
            iv = self.calculate_implied_volatility(
                option_price, strike_price, time_to_expiry,
                underlying_price, risk_free_rate, option_type
            )
            
            if iv is None or time_to_expiry <= 0:
                return None
            
            # Black-Scholes parameters
            d1 = (np.log(underlying_price / strike_price) + 
                  (risk_free_rate + 0.5 * iv ** 2) * time_to_expiry) / (iv * np.sqrt(time_to_expiry))
            d2 = d1 - iv * np.sqrt(time_to_expiry)
            
            # Calculate Greeks
            if option_type.lower() == 'call':
                delta = norm.cdf(d1)
                theta = (-underlying_price * norm.pdf(d1) * iv / (2 * np.sqrt(time_to_expiry)) -
                        risk_free_rate * strike_price * np.exp(-risk_free_rate * time_to_expiry) * norm.cdf(d2))
            else:  # put
                delta = -norm.cdf(-d1)
                theta = (-underlying_price * norm.pdf(d1) * iv / (2 * np.sqrt(time_to_expiry)) +
                        risk_free_rate * strike_price * np.exp(-risk_free_rate * time_to_expiry) * norm.cdf(-d2))
            
            gamma = norm.pdf(d1) / (underlying_price * iv * np.sqrt(time_to_expiry))
            vega = underlying_price * norm.pdf(d1) * np.sqrt(time_to_expiry) / 100
            
            # Convert theta to daily
            theta = theta / 365
            
            return OptionGreeks(
                delta=delta,
                gamma=gamma,
                theta=theta,
                vega=vega,
                implied_volatility=iv
            )
            
        except Exception as e:
            logger.error(f"Error calculating Greeks: {e}")
            return None

class RiskManager:
    """Risk management system"""
    
    def __init__(self, config: BotConfig):
        self.config = config
        self.daily_pnl = 0.0
        self.positions = {}
        
    def validate_position(self, position_data: Dict) -> Tuple[bool, str]:
        """Validate new position against risk limits"""
        
        # Check position size
        position_value = position_data.get('max_loss', 0)
        max_position_value = position_data.get('portfolio_value', 25000) * self.config.max_position_size_pct
        
        if position_value > max_position_value:
            return False, f"Position risk {position_value} exceeds limit {max_position_value}"
        
        # Check daily loss limit
        if abs(self.daily_pnl) > self.config.max_daily_loss:
            return False, f"Daily loss limit exceeded: {self.daily_pnl}"
        
        return True, "Position approved"
    
    def should_halt_trading(self) -> Tuple[bool, str]:
        """Check if trading should be halted"""
        
        if abs(self.daily_pnl) > self.config.max_daily_loss:
            return True, f"Daily loss limit exceeded: {self.daily_pnl}"
        
        return False, "Trading can continue"

class IronCondorStrategy:
    """Iron Condor options strategy implementation"""
    
    def __init__(self, config: BotConfig, data_manager: OptionsDataManager, risk_manager: RiskManager):
        self.config = config
        self.data_manager = data_manager
        self.risk_manager = risk_manager
        self.trading_client = TradingClient(
            api_key=config.alpaca_api_key,
            secret_key=config.alpaca_secret_key,
            paper=config.paper_trading
        )
        
    def find_options_chain(self, underlying_symbol: str) -> List[Dict]:
        """Find options chain for the underlying symbol"""
        try:
            # Calculate target expiration date
            target_exp = (datetime.now() + timedelta(days=self.config.target_dte)).date()
            
            # Get underlying price for strike filtering
            underlying_price = self.data_manager.get_underlying_price(underlying_symbol)
            strike_range = underlying_price * 0.15  # ±15% from current price
            
            request = GetOptionContractsRequest(
                underlying_symbols=[underlying_symbol],
                status=AssetStatus.ACTIVE,
                expiration_date=target_exp,
                strike_price_gte=str(underlying_price - strike_range),
                strike_price_lte=str(underlying_price + strike_range),
                style=ExerciseStyle.AMERICAN,
                limit=200
            )
            
            response = self.trading_client.get_option_contracts(request)
            contracts = []
            
            for contract in response.option_contracts:
                if contract.open_interest and float(contract.open_interest) > 50:
                    contracts.append({
                        'symbol': contract.symbol,
                        'underlying_symbol': contract.underlying_symbol,
                        'strike_price': float(contract.strike_price),
                        'expiration_date': str(contract.expiration_date),
                        'option_type': str(contract.type).lower(),
                        'open_interest': float(contract.open_interest)
                    })
            
            return contracts
            
        except Exception as e:
            logger.error(f"Error finding options chain: {e}")
            return []
    
    async def find_iron_condor_opportunity(self, underlying_symbol: str) -> Optional[Dict]:
        """Find Iron Condor trading opportunity"""
        
        try:
            underlying_price = self.data_manager.get_underlying_price(underlying_symbol)
            options_chain = self.find_options_chain(underlying_symbol)
            
            if not options_chain:
                logger.warning(f"No options chain found for {underlying_symbol}")
                return None
            
            # Separate calls and puts
            calls = [opt for opt in options_chain if opt['option_type'] == 'call']
            puts = [opt for opt in options_chain if opt['option_type'] == 'put']
            
            # Find short options (around target delta)
            short_call = await self.find_option_by_delta(calls, underlying_price, -self.config.target_delta, 'call')
            short_put = await self.find_option_by_delta(puts, underlying_price, -self.config.target_delta, 'put')
            
            if not short_call or not short_put:
                logger.warning("Could not find suitable short options")
                return None
            
            # Find long options (wing protection)
            long_call_strike = short_call['strike_price'] + self.config.wing_width
            long_put_strike = short_put['strike_price'] - self.config.wing_width
            
            long_call = self.find_option_by_strike(calls, long_call_strike)
            long_put = self.find_option_by_strike(puts, long_put_strike)
            
            if not long_call or not long_put:
                logger.warning("Could not find suitable long options for wings")
                return None
            
            # Get quotes for all legs
            quotes = {}
            for leg_name, option in [
                ('short_call', short_call), ('long_call', long_call),
                ('short_put', short_put), ('long_put', long_put)
            ]:
                quote = await self.data_manager.get_option_quote(option['symbol'])
                if not quote:
                    logger.warning(f"Could not get quote for {option['symbol']}")
                    return None
                quotes[leg_name] = quote
            
            # Calculate net credit
            net_credit = (
                quotes['short_call'].bid_price + quotes['short_put'].bid_price -
                quotes['long_call'].ask_price - quotes['long_put'].ask_price
            )
            
            # Calculate risk/reward metrics
            call_spread_width = long_call_strike - short_call['strike_price']
            put_spread_width = short_put['strike_price'] - long_put_strike
            max_spread_width = max(call_spread_width, put_spread_width)
            
            max_profit = net_credit * 100  # Per contract
            max_loss = (max_spread_width - net_credit) * 100
            
            # Validate minimum credit requirement
            if net_credit < (max_spread_width * self.config.min_credit_pct):
                logger.info(f"Credit {net_credit:.2f} below minimum threshold {max_spread_width * self.config.min_credit_pct:.2f}")
                return None
            
            opportunity = {
                'underlying_symbol': underlying_symbol,
                'underlying_price': underlying_price,
                'expiration_date': short_call['expiration_date'],
                'short_call': short_call,
                'long_call': long_call,
                'short_put': short_put,
                'long_put': long_put,
                'quotes': quotes,
                'net_credit': net_credit,
                'max_profit': max_profit,
                'max_loss': max_loss,
                'break_even_upper': short_call['strike_price'] + net_credit,
                'break_even_lower': short_put['strike_price'] - net_credit,
                'risk_reward_ratio': max_loss / max_profit if max_profit > 0 else float('inf')
            }
            
            return opportunity
            
        except Exception as e:
            logger.error(f"Error finding Iron Condor opportunity: {e}")
            return None
    
    async def find_option_by_delta(
        self, 
        options: List[Dict], 
        underlying_price: float, 
        target_delta: float, 
        option_type: str
    ) -> Optional[Dict]:
        """Find option closest to target delta"""
        
        best_option = None
        min_delta_diff = float('inf')
        
        for option in options:
            quote = await self.data_manager.get_option_quote(option['symbol'])
            if not quote:
                continue
            
            # Calculate time to expiry
            exp_date = datetime.strptime(option['expiration_date'], '%Y-%m-%d')
            tte = max((exp_date - datetime.now()).days / 365.0, 1/365)  # Minimum 1 day
            
            # Calculate delta
            greeks = self.data_manager.calculate_option_greeks(
                option_price=quote.mid_price,
                strike_price=option['strike_price'],
                time_to_expiry=tte,
                underlying_price=underlying_price,
                option_type=option_type
            )
            
            if greeks:
                delta_diff = abs(greeks.delta - target_delta)
                if delta_diff < min_delta_diff:
                    min_delta_diff = delta_diff
                    best_option = option
        
        return best_option
    
    def find_option_by_strike(self, options: List[Dict], target_strike: float) -> Optional[Dict]:
        """Find option with closest strike price"""
        best_option = None
        min_strike_diff = float('inf')
        
        for option in options:
            strike_diff = abs(option['strike_price'] - target_strike)
            if strike_diff < min_strike_diff:
                min_strike_diff = strike_diff
                best_option = option
        
        return best_option
    
    async def place_iron_condor_order(self, opportunity: Dict) -> Optional[str]:
        """Place Iron Condor order"""
        
        try:
            # Validate with risk manager
            portfolio_value = float(self.trading_client.get_account().portfolio_value)
            risk_data = {
                'max_loss': opportunity['max_loss'],
                'portfolio_value': portfolio_value
            }
            
            is_valid, reason = self.risk_manager.validate_position(risk_data)
            if not is_valid:
                logger.warning(f"Position rejected by risk manager: {reason}")
                return None
            
            # Build order legs
            legs = [
                OptionLegRequest(
                    symbol=opportunity['short_call']['symbol'],
                    side=OrderSide.SELL,
                    ratio_qty=1
                ),
                OptionLegRequest(
                    symbol=opportunity['long_call']['symbol'],
                    side=OrderSide.BUY,
                    ratio_qty=1
                ),
                OptionLegRequest(
                    symbol=opportunity['short_put']['symbol'],
                    side=OrderSide.SELL,
                    ratio_qty=1
                ),
                OptionLegRequest(
                    symbol=opportunity['long_put']['symbol'],
                    side=OrderSide.BUY,
                    ratio_qty=1
                )
            ]
            
            # Create market order for simplicity (in production, consider limit orders)
            order_request = MarketOrderRequest(
                qty=1,
                order_class=OrderClass.MLEG,
                time_in_force=TimeInForce.DAY,
                legs=legs
            )
            
            # Submit order
            order_response = self.trading_client.submit_order(order_request)
            
            logger.info(f"Iron Condor order placed: {order_response.id}")
            logger.info(f"Expected credit: ${opportunity['net_credit']:.2f}")
            logger.info(f"Max profit: ${opportunity['max_profit']:.2f}")
            logger.info(f"Max loss: ${opportunity['max_loss']:.2f}")
            
            return order_response.id
            
        except Exception as e:
            logger.error(f"Error placing Iron Condor order: {e}")
            return None

class OptionsBot:
    """Main options trading bot class"""
    
    def __init__(self, config: BotConfig):
        self.config = config
        self.data_manager = OptionsDataManager(config)
        self.risk_manager = RiskManager(config)
        self.strategy = IronCondorStrategy(config, self.data_manager, self.risk_manager)
        self.active_positions = {}
        self.running = False
        
    async def start(self):
        """Start the trading bot"""
        logger.info("Starting Options Trading Bot")
        logger.info(f"Paper Trading: {self.config.paper_trading}")
        logger.info(f"Target underlyings: SPY")
        
        self.running = True
        
        try:
            while self.running:
                await self.trading_cycle()
                await asyncio.sleep(self.config.check_interval)
                
        except KeyboardInterrupt:
            logger.info("Bot stopped by user")
        except Exception as e:
            logger.error(f"Bot error: {e}")
        finally:
            self.running = False
            logger.info("Options Trading Bot stopped")
    
    async def trading_cycle(self):
        """Execute one trading cycle"""
        try:
            # Check if trading should be halted
            should_halt, reason = self.risk_manager.should_halt_trading()
            if should_halt:
                logger.warning(f"Trading halted: {reason}")
                return
            
            # Look for new opportunities if no active positions
            if not self.active_positions:
                await self.scan_for_opportunities()
            
            # Monitor existing positions
            await self.monitor_positions()
            
        except Exception as e:
            logger.error(f"Error in trading cycle: {e}")
    
    async def scan_for_opportunities(self):
        """Scan for new trading opportunities"""
        try:
            # Focus on SPY for this example
            underlying_symbols = ['SPY']
            
            for symbol in underlying_symbols:
                logger.info(f"Scanning for opportunities in {symbol}")
                
                opportunity = await self.strategy.find_iron_condor_opportunity(symbol)
                
                if opportunity:
                    logger.info(f"Found Iron Condor opportunity in {symbol}")
                    logger.info(f"Net credit: ${opportunity['net_credit']:.2f}")
                    logger.info(f"Max profit: ${opportunity['max_profit']:.2f}")
                    logger.info(f"Max loss: ${opportunity['max_loss']:.2f}")
                    logger.info(f"Risk/Reward: {opportunity['risk_reward_ratio']:.2f}")
                    
                    # Place the order
                    order_id = await self.strategy.place_iron_condor_order(opportunity)
                    
                    if order_id:
                        # Track the position
                        position = IronCondorPosition(
                            underlying_symbol=symbol,
                            expiration_date=opportunity['expiration_date'],
                            short_call_symbol=opportunity['short_call']['symbol'],
                            long_call_symbol=opportunity['long_call']['symbol'],
                            short_put_symbol=opportunity['short_put']['symbol'],
                            long_put_symbol=opportunity['long_put']['symbol'],
                            net_credit=opportunity['net_credit'],
                            max_profit=opportunity['max_profit'],
                            max_loss=opportunity['max_loss'],
                            break_even_upper=opportunity['break_even_upper'],
                            break_even_lower=opportunity['break_even_lower'],
                            entry_time=datetime.now()
                        )
                        
                        self.active_positions[order_id] = position
                        logger.info(f"Added position to tracking: {order_id}")
                        
                        # Only trade one position at a time for this example
                        break
                
        except Exception as e:
            logger.error(f"Error scanning for opportunities: {e}")
    
    async def monitor_positions(self):
        """Monitor existing positions for exit conditions"""
        
        for order_id, position in list(self.active_positions.items()):
            try:
                # Check if position should be closed
                should_close, reason = await self.should_close_position(position)
                
                if should_close:
                    logger.info(f"Closing position {order_id}: {reason}")
                    success = await self.close_position(order_id, position)
                    
                    if success:
                        del self.active_positions[order_id]
                        logger.info(f"Position {order_id} closed successfully")
                    
            except Exception as e:
                logger.error(f"Error monitoring position {order_id}: {e}")
    
    async def should_close_position(self, position: IronCondorPosition) -> Tuple[bool, str]:
        """Determine if position should be closed"""
        
        try:
            # Check time to expiration
            exp_date = datetime.strptime(position.expiration_date, '%Y-%m-%d')
            days_to_expiry = (exp_date - datetime.now()).days
            
            if days_to_expiry <= 7:
                return True, f"Close to expiration: {days_to_expiry} days"
            
            # Check if profit target reached (50% of max profit)
            current_value = await self.get_position_value(position)
            if current_value is not None:
                profit_pct = (position.net_credit * 100 - current_value) / (position.net_credit * 100)
                
                if profit_pct >= 0.5:  # 50% profit target
                    return True, f"Profit target reached: {profit_pct:.2%}"
                
                if profit_pct <= -1.0:  # 100% loss (double the credit received)
                    return True, f"Stop loss triggered: {profit_pct:.2%}"
            
            # Check underlying price vs break-even points
            underlying_price = self.data_manager.get_underlying_price(position.underlying_symbol)
            
            if underlying_price >= position.break_even_upper:
                return True, f"Price above upper break-even: {underlying_price} >= {position.break_even_upper}"
            
            if underlying_price <= position.break_even_lower:
                return True, f"Price below lower break-even: {underlying_price} <= {position.break_even_lower}"
            
            return False, "Position within parameters"
            
        except Exception as e:
            logger.error(f"Error checking position exit conditions: {e}")
            return True, f"Error in position check: {e}"
    
    async def get_position_value(self, position: IronCondorPosition) -> Optional[float]:
        """Get current value of the Iron Condor position"""
        
        try:
            total_value = 0
            
            # Get quotes for all legs
            symbols = [
                position.short_call_symbol,
                position.long_call_symbol,
                position.short_put_symbol,
                position.long_put_symbol
            ]
            
            sides = ['sell', 'buy', 'sell', 'buy']  # Iron Condor structure
            
            for symbol, side in zip(symbols, sides):
                quote = await self.data_manager.get_option_quote(symbol)
                if quote:
                    # Use bid for selling, ask for buying to be conservative
                    price = quote.bid_price if side == 'sell' else quote.ask_price
                    total_value += price * (1 if side == 'sell' else -1)
            
            return total_value * 100  # Convert to total dollar value
            
        except Exception as e:
            logger.error(f"Error getting position value: {e}")
            return None
    
    async def close_position(self, order_id: str, position: IronCondorPosition) -> bool:
        """Close an Iron Condor position"""
        
        try:
            # Create closing order (reverse the original order)
            legs = [
                OptionLegRequest(
                    symbol=position.short_call_symbol,
                    side=OrderSide.BUY,  # Buy back the short call
                    ratio_qty=1
                ),
                OptionLegRequest(
                    symbol=position.long_call_symbol,
                    side=OrderSide.SELL,  # Sell the long call
                    ratio_qty=1
                ),
                OptionLegRequest(
                    symbol=position.short_put_symbol,
                    side=OrderSide.BUY,  # Buy back the short put
                    ratio_qty=1
                ),
                OptionLegRequest(
                    symbol=position.long_put_symbol,
                    side=OrderSide.SELL,  # Sell the long put
                    ratio_qty=1
                )
            ]
            
            # Create market order to close
            order_request = MarketOrderRequest(
                qty=1,
                order_class=OrderClass.MLEG,
                time_in_force=TimeInForce.DAY,
                legs=legs
            )
            
            # Submit closing order
            order_response = self.strategy.trading_client.submit_order(order_request)
            
            logger.info(f"Closing order placed: {order_response.id}")
            return True
            
        except Exception as e:
            logger.error(f"Error closing position: {e}")
            return False
    
    def stop(self):
        """Stop the trading bot"""
        self.running = False

def main():
    """Main function to run the options trading bot"""
    
    # Load configuration
    config = BotConfig(
        alpaca_api_key=os.getenv('ALPACA_API_KEY'),
        alpaca_secret_key=os.getenv('ALPACA_SECRET_KEY'),
        paper_trading=True,  # Always start with paper trading
        max_daily_loss=500.0,  # Conservative for demo
        max_position_size_pct=0.02,  # 2% of portfolio
        target_dte=30,
        check_interval=300  # Check every 5 minutes
    )
    
    # Validate configuration
    if not config.alpaca_api_key or not config.alpaca_secret_key:
        logger.error("Missing Alpaca API credentials. Please set ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables.")
        return
    
    # Create and start the bot
    bot = OptionsBot(config)
    
    try:
        # Run the bot
        asyncio.run(bot.start())
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
        bot.stop()

if __name__ == "__main__":
    main()
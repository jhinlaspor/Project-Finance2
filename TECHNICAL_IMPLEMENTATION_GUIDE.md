# Technical Implementation Guide: Options Trading Bots with Alpaca

## Table of Contents

1. [Prerequisites & Setup](#prerequisites--setup)
2. [Alpaca API Integration](#alpaca-api-integration)
3. [Options Data Management](#options-data-management)
4. [Strategy Implementation Examples](#strategy-implementation-examples)
5. [Risk Management Systems](#risk-management-systems)
6. [Order Execution Framework](#order-execution-framework)
7. [Monitoring & Logging](#monitoring--logging)
8. [Testing & Validation](#testing--validation)
9. [Deployment Guide](#deployment-guide)
10. [Troubleshooting](#troubleshooting)

## Prerequisites & Setup

### 1. Environment Requirements

```bash
# Python version
python --version  # Requires Python 3.8+

# Create virtual environment
python -m venv alpaca_options_env
source alpaca_options_env/bin/activate  # On Windows: alpaca_options_env\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

### 2. Requirements File

```txt
# requirements.txt
alpaca-py==0.23.0
pandas==2.2.2
numpy==1.26.4
scipy==1.10.0
ta==0.11.0
python-dotenv==1.0.0
websockets==11.0.3
structlog==24.1.0
pytest==8.2.0
aiohttp==3.9.0
asyncio-mqtt==0.16.1
redis==5.0.0
pydantic==2.5.0
fastapi==0.104.0
uvicorn==0.24.0
```

### 3. Environment Variables

```bash
# .env file
ALPACA_API_KEY=your_paper_trading_api_key
ALPACA_SECRET_KEY=your_paper_trading_secret_key
ALPACA_BASE_URL=https://paper-api.alpaca.markets
POLYGON_API_KEY=your_polygon_api_key_optional

# Risk Management
MAX_DAILY_LOSS=2000
MAX_POSITION_SIZE=0.05
MAX_PORTFOLIO_DELTA=1000

# Trading Configuration
TRADING_MODE=paper
LOG_LEVEL=INFO
REDIS_URL=redis://localhost:6379
```

### 4. Project Structure Setup

```bash
mkdir alpaca_options_bot
cd alpaca_options_bot

# Create directory structure
mkdir -p {core,strategies,utils,config,tests,logs,data}
touch {core,strategies,utils,config,tests}/__init__.py

# Create main files
touch main.py requirements.txt .env .gitignore README.md
```

## Alpaca API Integration

### 1. Basic API Client Setup

```python
# core/alpaca_client.py
import os
import asyncio
from typing import Optional, List, Dict, Any
from alpaca.trading.client import TradingClient
from alpaca.data.historical.option import OptionHistoricalDataClient
from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.trading.stream import TradingStream
from alpaca.data.live.option import OptionDataStream
import structlog

logger = structlog.get_logger(__name__)

class AlpacaOptionsClient:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.trading_client = TradingClient(
            api_key=config['api_key'],
            secret_key=config['secret_key'],
            paper=config.get('paper', True),
            url_override=config.get('base_url')
        )
        
        self.option_data_client = OptionHistoricalDataClient(
            api_key=config['api_key'],
            secret_key=config['secret_key']
        )
        
        self.stock_data_client = StockHistoricalDataClient(
            api_key=config['api_key'],
            secret_key=config['secret_key']
        )
        
        self.trade_stream = None
        self.option_stream = None
        
    async def initialize_streams(self):
        """Initialize real-time data streams"""
        try:
            self.trade_stream = TradingStream(
                api_key=self.config['api_key'],
                secret_key=self.config['secret_key'],
                paper=self.config.get('paper', True)
            )
            
            self.option_stream = OptionDataStream(
                api_key=self.config['api_key'],
                secret_key=self.config['secret_key']
            )
            
            logger.info("Alpaca streams initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize streams: {e}")
            raise
    
    def get_account_info(self):
        """Get account information"""
        try:
            account = self.trading_client.get_account()
            return {
                'buying_power': float(account.buying_power),
                'portfolio_value': float(account.portfolio_value),
                'cash': float(account.cash),
                'day_trade_count': account.day_trade_count,
                'pattern_day_trader': account.pattern_day_trader
            }
        except Exception as e:
            logger.error(f"Failed to get account info: {e}")
            raise
```

### 2. Options Contract Discovery

```python
# core/options_discovery.py
from datetime import datetime, timedelta
from typing import List, Optional, Dict
from alpaca.trading.requests import GetOptionContractsRequest
from alpaca.trading.enums import AssetStatus, ContractType, ExerciseStyle
import pandas as pd

class OptionsContractDiscovery:
    def __init__(self, alpaca_client):
        self.client = alpaca_client
        
    def find_options_chain(
        self,
        underlying_symbol: str,
        expiration_date: Optional[str] = None,
        min_days_to_expiry: int = 7,
        max_days_to_expiry: int = 45,
        option_type: Optional[ContractType] = None,
        min_open_interest: int = 100,
        strike_range_pct: float = 0.20
    ) -> List[Dict]:
        """Find options contracts based on criteria"""
        
        try:
            # Get underlying price for strike filtering
            underlying_price = self.get_underlying_price(underlying_symbol)
            
            # Calculate strike range
            min_strike = underlying_price * (1 - strike_range_pct)
            max_strike = underlying_price * (1 + strike_range_pct)
            
            # Set expiration date range if not specified
            if not expiration_date:
                today = datetime.now().date()
                min_exp = today + timedelta(days=min_days_to_expiry)
                max_exp = today + timedelta(days=max_days_to_expiry)
            else:
                min_exp = max_exp = datetime.strptime(expiration_date, '%Y-%m-%d').date()
            
            # Build request
            request = GetOptionContractsRequest(
                underlying_symbols=[underlying_symbol],
                status=AssetStatus.ACTIVE,
                expiration_date_gte=min_exp,
                expiration_date_lte=max_exp,
                strike_price_gte=str(min_strike),
                strike_price_lte=str(max_strike),
                type=option_type,
                style=ExerciseStyle.AMERICAN,
                limit=1000
            )
            
            # Get contracts
            response = self.client.trading_client.get_option_contracts(request)
            contracts = response.option_contracts
            
            # Filter by open interest and format
            filtered_contracts = []
            for contract in contracts:
                if (contract.open_interest and 
                    float(contract.open_interest) >= min_open_interest):
                    
                    contract_data = {
                        'symbol': contract.symbol,
                        'underlying_symbol': contract.underlying_symbol,
                        'strike_price': float(contract.strike_price),
                        'expiration_date': contract.expiration_date,
                        'option_type': str(contract.type),
                        'open_interest': float(contract.open_interest),
                        'size': float(contract.size),
                        'id': contract.id
                    }
                    filtered_contracts.append(contract_data)
            
            return filtered_contracts
            
        except Exception as e:
            logger.error(f"Error finding options chain for {underlying_symbol}: {e}")
            return []
    
    def get_underlying_price(self, symbol: str) -> float:
        """Get current underlying asset price"""
        try:
            from alpaca.data.requests import StockLatestTradeRequest
            
            request = StockLatestTradeRequest(symbol_or_symbols=symbol)
            response = self.client.stock_data_client.get_stock_latest_trade(request)
            return float(response[symbol].price)
            
        except Exception as e:
            logger.error(f"Error getting price for {symbol}: {e}")
            raise
```

## Options Data Management

### 1. Real-time Options Data Handler

```python
# core/options_data_manager.py
import asyncio
from typing import Dict, List, Callable, Optional
from dataclasses import dataclass
from datetime import datetime
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq

@dataclass
class OptionQuote:
    symbol: str
    bid_price: float
    ask_price: float
    bid_size: int
    ask_size: int
    timestamp: datetime
    
@dataclass
class OptionGreeks:
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float
    implied_volatility: float

class OptionsDataManager:
    def __init__(self, alpaca_client):
        self.client = alpaca_client
        self.quote_cache = {}
        self.greeks_cache = {}
        self.subscribers = []
        
    async def get_option_quote(self, symbol: str) -> Optional[OptionQuote]:
        """Get latest option quote"""
        try:
            from alpaca.data.requests import OptionLatestQuoteRequest
            
            request = OptionLatestQuoteRequest(symbol_or_symbols=[symbol])
            response = self.client.option_data_client.get_option_latest_quote(request)
            
            if symbol in response:
                quote_data = response[symbol]
                quote = OptionQuote(
                    symbol=symbol,
                    bid_price=float(quote_data.bid_price),
                    ask_price=float(quote_data.ask_price),
                    bid_size=int(quote_data.bid_size),
                    ask_size=int(quote_data.ask_size),
                    timestamp=quote_data.timestamp
                )
                
                # Cache the quote
                self.quote_cache[symbol] = quote
                return quote
                
        except Exception as e:
            logger.error(f"Error getting quote for {symbol}: {e}")
            return None
    
    def calculate_option_greeks(
        self,
        option_price: float,
        strike_price: float,
        time_to_expiry: float,
        underlying_price: float,
        risk_free_rate: float = 0.045,
        option_type: str = 'call'
    ) -> OptionGreeks:
        """Calculate option Greeks using Black-Scholes"""
        
        try:
            # Calculate implied volatility first
            iv = self.calculate_implied_volatility(
                option_price, strike_price, time_to_expiry,
                underlying_price, risk_free_rate, option_type
            )
            
            if iv is None:
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
            rho = (strike_price * time_to_expiry * np.exp(-risk_free_rate * time_to_expiry) *
                   (norm.cdf(d2) if option_type.lower() == 'call' else -norm.cdf(-d2))) / 100
            
            # Adjust theta for daily decay
            theta = theta / 365
            
            return OptionGreeks(
                delta=delta,
                gamma=gamma,
                theta=theta,
                vega=vega,
                rho=rho,
                implied_volatility=iv
            )
            
        except Exception as e:
            logger.error(f"Error calculating Greeks: {e}")
            return None
    
    def calculate_implied_volatility(
        self,
        option_price: float,
        strike_price: float,
        time_to_expiry: float,
        underlying_price: float,
        risk_free_rate: float,
        option_type: str,
        max_iterations: int = 100,
        tolerance: float = 1e-6
    ) -> Optional[float]:
        """Calculate implied volatility using Brent's method"""
        
        def black_scholes_price(volatility):
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
            iv = brentq(black_scholes_price, 0.001, 5.0, maxiter=max_iterations, xtol=tolerance)
            return iv
            
        except (ValueError, RuntimeError):
            # Fallback to simple iteration if Brent's method fails
            try:
                for vol in np.arange(0.01, 3.0, 0.01):
                    if abs(black_scholes_price(vol)) < tolerance:
                        return vol
            except:
                pass
            
            logger.warning(f"Could not calculate IV for option price {option_price}")
            return None
    
    async def subscribe_to_option_stream(self, symbols: List[str], callback: Callable):
        """Subscribe to real-time option data stream"""
        try:
            async def quote_handler(data):
                quote = OptionQuote(
                    symbol=data.symbol,
                    bid_price=data.bid_price,
                    ask_price=data.ask_price,
                    bid_size=data.bid_size,
                    ask_size=data.ask_size,
                    timestamp=data.timestamp
                )
                await callback(quote)
            
            await self.client.option_stream.subscribe_quotes(quote_handler, *symbols)
            logger.info(f"Subscribed to option quotes for {symbols}")
            
        except Exception as e:
            logger.error(f"Error subscribing to option stream: {e}")
```

## Strategy Implementation Examples

### 1. Iron Condor Strategy

```python
# strategies/iron_condor.py
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np

@dataclass
class IronCondorLeg:
    symbol: str
    option_type: str  # 'call' or 'put'
    strike_price: float
    action: str  # 'buy' or 'sell'
    quantity: int
    premium: float

@dataclass
class IronCondorPosition:
    underlying_symbol: str
    expiration_date: datetime
    short_call: IronCondorLeg
    long_call: IronCondorLeg
    short_put: IronCondorLeg
    long_put: IronCondorLeg
    net_credit: float
    max_profit: float
    max_loss: float
    created_at: datetime

class IronCondorStrategy:
    def __init__(self, data_manager, risk_manager):
        self.data_manager = data_manager
        self.risk_manager = risk_manager
        self.positions = {}
        
    async def find_iron_condor_opportunity(
        self,
        underlying_symbol: str,
        target_dte: int = 30,
        target_delta_short_call: float = -0.20,
        target_delta_short_put: float = -0.20,
        min_credit_percentage: float = 0.33,
        wing_width: float = 5.0
    ) -> Optional[Dict]:
        """Find Iron Condor trading opportunity"""
        
        try:
            # Get underlying price
            underlying_price = self.data_manager.get_underlying_price(underlying_symbol)
            
            # Find options chain
            target_exp = (datetime.now() + timedelta(days=target_dte)).strftime('%Y-%m-%d')
            options_chain = self.data_manager.find_options_chain(
                underlying_symbol=underlying_symbol,
                expiration_date=target_exp,
                strike_range_pct=0.15
            )
            
            if not options_chain:
                return None
            
            # Separate calls and puts
            calls = [opt for opt in options_chain if opt['option_type'] == 'call']
            puts = [opt for opt in options_chain if opt['option_type'] == 'put']
            
            # Find short call (OTM call with target delta)
            short_call = await self.find_option_by_delta(
                calls, underlying_price, target_delta_short_call, 'call'
            )
            
            # Find short put (OTM put with target delta)
            short_put = await self.find_option_by_delta(
                puts, underlying_price, target_delta_short_put, 'put'
            )
            
            if not short_call or not short_put:
                return None
            
            # Find long call (further OTM)
            long_call_strike = short_call['strike_price'] + wing_width
            long_call = self.find_option_by_strike(calls, long_call_strike)
            
            # Find long put (further OTM)
            long_put_strike = short_put['strike_price'] - wing_width
            long_put = self.find_option_by_strike(puts, long_put_strike)
            
            if not long_call or not long_put:
                return None
            
            # Get current quotes for all legs
            short_call_quote = await self.data_manager.get_option_quote(short_call['symbol'])
            long_call_quote = await self.data_manager.get_option_quote(long_call['symbol'])
            short_put_quote = await self.data_manager.get_option_quote(short_put['symbol'])
            long_put_quote = await self.data_manager.get_option_quote(long_put['symbol'])
            
            if not all([short_call_quote, long_call_quote, short_put_quote, long_put_quote]):
                return None
            
            # Calculate net credit
            net_credit = (
                (short_call_quote.bid_price + short_put_quote.bid_price) -
                (long_call_quote.ask_price + long_put_quote.ask_price)
            )
            
            # Calculate max profit/loss
            call_spread_width = long_call_strike - short_call['strike_price']
            put_spread_width = short_put['strike_price'] - long_put_strike
            max_spread_width = max(call_spread_width, put_spread_width)
            
            max_profit = net_credit * 100  # Per contract
            max_loss = (max_spread_width - net_credit) * 100
            
            # Check minimum credit requirement
            if net_credit < (max_spread_width * min_credit_percentage):
                return None
            
            # Build opportunity structure
            opportunity = {
                'underlying_symbol': underlying_symbol,
                'underlying_price': underlying_price,
                'expiration_date': target_exp,
                'legs': [
                    {'symbol': short_call['symbol'], 'action': 'sell', 'type': 'call', 
                     'strike': short_call['strike_price'], 'premium': short_call_quote.bid_price},
                    {'symbol': long_call['symbol'], 'action': 'buy', 'type': 'call',
                     'strike': long_call_strike, 'premium': long_call_quote.ask_price},
                    {'symbol': short_put['symbol'], 'action': 'sell', 'type': 'put',
                     'strike': short_put['strike_price'], 'premium': short_put_quote.bid_price},
                    {'symbol': long_put['symbol'], 'action': 'buy', 'type': 'put',
                     'strike': long_put_strike, 'premium': long_put_quote.ask_price}
                ],
                'net_credit': net_credit,
                'max_profit': max_profit,
                'max_loss': max_loss,
                'break_even_upper': short_call['strike_price'] + net_credit,
                'break_even_lower': short_put['strike_price'] - net_credit
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
            
            mid_price = (quote.bid_price + quote.ask_price) / 2
            
            # Calculate time to expiry
            exp_date = datetime.strptime(option['expiration_date'], '%Y-%m-%d')
            tte = (exp_date - datetime.now()).days / 365.0
            
            # Calculate Greeks
            greeks = self.data_manager.calculate_option_greeks(
                option_price=mid_price,
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
        """Find option with specific strike price"""
        for option in options:
            if abs(option['strike_price'] - target_strike) < 0.01:
                return option
        return None
```

### 2. Gamma Scalping Strategy

```python
# strategies/gamma_scalping.py
import asyncio
from typing import Dict, Optional
from datetime import datetime, timedelta

class GammaScalpingStrategy:
    def __init__(self, data_manager, risk_manager, order_manager):
        self.data_manager = data_manager
        self.risk_manager = risk_manager
        self.order_manager = order_manager
        self.positions = {}
        self.hedge_threshold = 100  # Delta threshold for hedging
        
    async def initialize_position(
        self,
        underlying_symbol: str,
        option_quantity: int = 10,
        target_dte: int = 30
    ) -> bool:
        """Initialize gamma scalping position with long straddle"""
        
        try:
            # Find ATM straddle
            straddle = await self.find_atm_straddle(underlying_symbol, target_dte)
            if not straddle:
                logger.error(f"Could not find suitable straddle for {underlying_symbol}")
                return False
            
            # Place straddle order
            order_legs = [
                {
                    'symbol': straddle['call']['symbol'],
                    'side': 'buy',
                    'quantity': option_quantity
                },
                {
                    'symbol': straddle['put']['symbol'],
                    'side': 'buy',
                    'quantity': option_quantity
                }
            ]
            
            order_result = await self.order_manager.place_multi_leg_order(order_legs)
            if not order_result:
                return False
            
            # Store position
            self.positions[underlying_symbol] = {
                'type': 'gamma_scalp',
                'call_symbol': straddle['call']['symbol'],
                'put_symbol': straddle['put']['symbol'],
                'quantity': option_quantity,
                'underlying_hedge': 0,
                'created_at': datetime.now(),
                'last_hedge': datetime.now()
            }
            
            logger.info(f"Initialized gamma scalping position for {underlying_symbol}")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing gamma scalping position: {e}")
            return False
    
    async def monitor_and_hedge(self, underlying_symbol: str):
        """Monitor position and perform delta hedging"""
        
        if underlying_symbol not in self.positions:
            return
        
        position = self.positions[underlying_symbol]
        
        try:
            # Calculate current portfolio delta
            current_delta = await self.calculate_portfolio_delta(position)
            
            # Check if hedging is needed
            if abs(current_delta) > self.hedge_threshold:
                await self.perform_delta_hedge(underlying_symbol, current_delta)
                
        except Exception as e:
            logger.error(f"Error monitoring gamma scalping position: {e}")
    
    async def calculate_portfolio_delta(self, position: Dict) -> float:
        """Calculate current delta of the gamma scalping position"""
        
        try:
            underlying_price = self.data_manager.get_underlying_price(
                position['call_symbol'].split('_')[0]
            )
            
            total_delta = 0
            
            # Get call delta
            call_quote = await self.data_manager.get_option_quote(position['call_symbol'])
            if call_quote:
                call_greeks = self.calculate_option_greeks_from_quote(
                    call_quote, underlying_price, 'call'
                )
                if call_greeks:
                    total_delta += call_greeks.delta * position['quantity']
            
            # Get put delta
            put_quote = await self.data_manager.get_option_quote(position['put_symbol'])
            if put_quote:
                put_greeks = self.calculate_option_greeks_from_quote(
                    put_quote, underlying_price, 'put'
                )
                if put_greeks:
                    total_delta += put_greeks.delta * position['quantity']
            
            # Add underlying hedge delta
            total_delta += position['underlying_hedge']
            
            return total_delta
            
        except Exception as e:
            logger.error(f"Error calculating portfolio delta: {e}")
            return 0
    
    async def perform_delta_hedge(self, underlying_symbol: str, current_delta: float):
        """Perform delta hedging by trading underlying"""
        
        try:
            position = self.positions[underlying_symbol]
            
            # Calculate hedge quantity (opposite of delta)
            hedge_quantity = -int(current_delta)
            
            # Place hedge order
            if hedge_quantity != 0:
                order_result = await self.order_manager.place_stock_order(
                    symbol=underlying_symbol,
                    quantity=abs(hedge_quantity),
                    side='buy' if hedge_quantity > 0 else 'sell'
                )
                
                if order_result:
                    # Update position
                    position['underlying_hedge'] += hedge_quantity
                    position['last_hedge'] = datetime.now()
                    
                    logger.info(f"Performed delta hedge: {hedge_quantity} shares of {underlying_symbol}")
                    
        except Exception as e:
            logger.error(f"Error performing delta hedge: {e}")
    
    async def find_atm_straddle(self, underlying_symbol: str, target_dte: int) -> Optional[Dict]:
        """Find at-the-money straddle for gamma scalping"""
        
        try:
            underlying_price = self.data_manager.get_underlying_price(underlying_symbol)
            
            # Get options chain
            target_exp = (datetime.now() + timedelta(days=target_dte)).strftime('%Y-%m-%d')
            options_chain = self.data_manager.find_options_chain(
                underlying_symbol=underlying_symbol,
                expiration_date=target_exp,
                strike_range_pct=0.05  # Narrow range around ATM
            )
            
            if not options_chain:
                return None
            
            # Find ATM strike
            atm_strike = min(options_chain, 
                           key=lambda x: abs(x['strike_price'] - underlying_price))['strike_price']
            
            # Find call and put at ATM strike
            call = next((opt for opt in options_chain 
                        if opt['strike_price'] == atm_strike and opt['option_type'] == 'call'), None)
            put = next((opt for opt in options_chain 
                       if opt['strike_price'] == atm_strike and opt['option_type'] == 'put'), None)
            
            if call and put:
                return {
                    'call': call,
                    'put': put,
                    'strike': atm_strike,
                    'underlying_price': underlying_price
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error finding ATM straddle: {e}")
            return None
```

## Risk Management Systems

### 1. Comprehensive Risk Manager

```python
# core/risk_manager.py
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, date
import numpy as np

@dataclass
class RiskLimits:
    max_portfolio_delta: float = 1000
    max_portfolio_gamma: float = 500
    max_portfolio_theta: float = -100
    max_position_size_pct: float = 0.05
    max_daily_loss: float = 2000
    max_option_allocation_pct: float = 0.30
    max_single_trade_risk: float = 500
    max_concentration_pct: float = 0.15

@dataclass
class PositionRisk:
    symbol: str
    position_value: float
    delta: float
    gamma: float
    theta: float
    vega: float
    time_to_expiry: float
    assignment_risk: float

class RiskManager:
    def __init__(self, config: Dict):
        self.limits = RiskLimits(**config.get('risk_limits', {}))
        self.portfolio_value = 0
        self.daily_pnl = 0
        self.positions = {}
        self.trade_history = []
        
    def update_portfolio_value(self, value: float):
        """Update current portfolio value"""
        self.portfolio_value = value
    
    def validate_new_position(self, position_data: Dict) -> Tuple[bool, str]:
        """Validate if new position meets risk criteria"""
        
        try:
            # Check position size
            position_value = position_data.get('value', 0)
            max_position_value = self.portfolio_value * self.limits.max_position_size_pct
            
            if position_value > max_position_value:
                return False, f"Position size {position_value} exceeds limit {max_position_value}"
            
            # Check single trade risk
            trade_risk = position_data.get('max_loss', 0)
            if trade_risk > self.limits.max_single_trade_risk:
                return False, f"Trade risk {trade_risk} exceeds limit {self.limits.max_single_trade_risk}"
            
            # Check options allocation
            current_options_value = self.calculate_total_options_value()
            new_options_allocation = (current_options_value + position_value) / self.portfolio_value
            
            if new_options_allocation > self.limits.max_option_allocation_pct:
                return False, f"Options allocation {new_options_allocation:.2%} exceeds limit {self.limits.max_option_allocation_pct:.2%}"
            
            # Check concentration risk
            underlying = position_data.get('underlying_symbol', '')
            current_underlying_exposure = self.calculate_underlying_exposure(underlying)
            new_concentration = (current_underlying_exposure + position_value) / self.portfolio_value
            
            if new_concentration > self.limits.max_concentration_pct:
                return False, f"Concentration in {underlying} would exceed limit"
            
            # Check portfolio Greeks after adding position
            portfolio_greeks = self.calculate_portfolio_greeks()
            position_greeks = position_data.get('greeks', {})
            
            new_delta = portfolio_greeks.get('delta', 0) + position_greeks.get('delta', 0)
            new_gamma = portfolio_greeks.get('gamma', 0) + position_greeks.get('gamma', 0)
            new_theta = portfolio_greeks.get('theta', 0) + position_greeks.get('theta', 0)
            
            if abs(new_delta) > self.limits.max_portfolio_delta:
                return False, f"Portfolio delta {new_delta} would exceed limit {self.limits.max_portfolio_delta}"
            
            if abs(new_gamma) > self.limits.max_portfolio_gamma:
                return False, f"Portfolio gamma {new_gamma} would exceed limit {self.limits.max_portfolio_gamma}"
            
            if new_theta < self.limits.max_portfolio_theta:
                return False, f"Portfolio theta {new_theta} would exceed limit {self.limits.max_portfolio_theta}"
            
            return True, "Position approved"
            
        except Exception as e:
            logger.error(f"Error validating position: {e}")
            return False, f"Validation error: {str(e)}"
    
    def check_daily_loss_limit(self) -> bool:
        """Check if daily loss limit is exceeded"""
        return self.daily_pnl < -self.limits.max_daily_loss
    
    def calculate_portfolio_greeks(self) -> Dict[str, float]:
        """Calculate aggregate portfolio Greeks"""
        
        total_delta = 0
        total_gamma = 0
        total_theta = 0
        total_vega = 0
        
        for position in self.positions.values():
            greeks = position.get('greeks', {})
            quantity = position.get('quantity', 0)
            
            total_delta += greeks.get('delta', 0) * quantity
            total_gamma += greeks.get('gamma', 0) * quantity
            total_theta += greeks.get('theta', 0) * quantity
            total_vega += greeks.get('vega', 0) * quantity
        
        return {
            'delta': total_delta,
            'gamma': total_gamma,
            'theta': total_theta,
            'vega': total_vega
        }
    
    def calculate_assignment_risk(self) -> List[Dict]:
        """Calculate assignment risk for short options"""
        
        assignment_risks = []
        today = date.today()
        
        for symbol, position in self.positions.items():
            if position.get('side') == 'short':
                expiry = position.get('expiration_date')
                strike = position.get('strike_price')
                option_type = position.get('option_type')
                underlying_price = position.get('underlying_price', 0)
                
                # Check if option is ITM
                if option_type == 'call' and underlying_price > strike:
                    itm_amount = underlying_price - strike
                    assignment_risks.append({
                        'symbol': symbol,
                        'type': 'call_assignment',
                        'itm_amount': itm_amount,
                        'expiry': expiry,
                        'probability': self.calculate_assignment_probability(itm_amount, expiry)
                    })
                elif option_type == 'put' and underlying_price < strike:
                    itm_amount = strike - underlying_price
                    assignment_risks.append({
                        'symbol': symbol,
                        'type': 'put_assignment',
                        'itm_amount': itm_amount,
                        'expiry': expiry,
                        'probability': self.calculate_assignment_probability(itm_amount, expiry)
                    })
        
        return assignment_risks
    
    def calculate_assignment_probability(self, itm_amount: float, expiry: date) -> float:
        """Calculate probability of assignment based on ITM amount and time to expiry"""
        
        days_to_expiry = (expiry - date.today()).days
        
        # Simple heuristic: higher ITM amount and closer to expiry = higher probability
        if days_to_expiry <= 0:
            return 0.95 if itm_amount > 0.01 else 0.05
        elif days_to_expiry <= 7:
            return min(0.8, 0.1 + (itm_amount * 5))
        else:
            return min(0.3, 0.05 + (itm_amount * 2))
    
    def calculate_total_options_value(self) -> float:
        """Calculate total value of options positions"""
        return sum(pos.get('value', 0) for pos in self.positions.values() 
                  if pos.get('asset_type') == 'option')
    
    def calculate_underlying_exposure(self, underlying_symbol: str) -> float:
        """Calculate total exposure to specific underlying"""
        return sum(pos.get('value', 0) for pos in self.positions.values() 
                  if pos.get('underlying_symbol') == underlying_symbol)
    
    def should_halt_trading(self) -> Tuple[bool, str]:
        """Determine if trading should be halted"""
        
        # Check daily loss limit
        if self.check_daily_loss_limit():
            return True, f"Daily loss limit exceeded: ${self.daily_pnl}"
        
        # Check portfolio Greeks limits
        portfolio_greeks = self.calculate_portfolio_greeks()
        
        if abs(portfolio_greeks['delta']) > self.limits.max_portfolio_delta:
            return True, f"Portfolio delta limit exceeded: {portfolio_greeks['delta']}"
        
        if abs(portfolio_greeks['gamma']) > self.limits.max_portfolio_gamma:
            return True, f"Portfolio gamma limit exceeded: {portfolio_greeks['gamma']}"
        
        # Check assignment risks
        assignment_risks = self.calculate_assignment_risk()
        high_risk_assignments = [risk for risk in assignment_risks if risk['probability'] > 0.7]
        
        if len(high_risk_assignments) > 3:
            return True, f"Too many high-risk assignments: {len(high_risk_assignments)}"
        
        return False, "Trading can continue"
```

## Order Execution Framework

### 1. Multi-leg Order Manager

```python
# core/order_manager.py
import asyncio
from typing import Dict, List, Optional, Union
from enum import Enum
from dataclasses import dataclass
from datetime import datetime
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest, OptionLegRequest
from alpaca.trading.enums import OrderSide, OrderClass, TimeInForce

class OrderStatus(Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    FAILED = "failed"

@dataclass
class OrderLeg:
    symbol: str
    side: str  # 'buy' or 'sell'
    quantity: int
    order_type: str = 'market'
    limit_price: Optional[float] = None

@dataclass
class OrderRequest:
    legs: List[OrderLeg]
    order_class: str = 'mleg'
    time_in_force: str = 'day'
    limit_price: Optional[float] = None

class OrderManager:
    def __init__(self, alpaca_client, risk_manager):
        self.client = alpaca_client
        self.risk_manager = risk_manager
        self.pending_orders = {}
        self.order_history = []
        
    async def place_multi_leg_order(
        self,
        order_request: OrderRequest,
        validate_risk: bool = True
    ) -> Optional[Dict]:
        """Place multi-leg options order"""
        
        try:
            # Risk validation
            if validate_risk:
                is_valid, reason = self.risk_manager.validate_order(order_request)
                if not is_valid:
                    logger.warning(f"Order rejected by risk manager: {reason}")
                    return None
            
            # Build Alpaca order legs
            alpaca_legs = []
            for leg in order_request.legs:
                alpaca_leg = OptionLegRequest(
                    symbol=leg.symbol,
                    side=OrderSide.BUY if leg.side.lower() == 'buy' else OrderSide.SELL,
                    ratio_qty=leg.quantity
                )
                alpaca_legs.append(alpaca_leg)
            
            # Create order request
            if order_request.limit_price:
                alpaca_order = LimitOrderRequest(
                    qty=1,  # For multi-leg, qty represents number of spreads
                    order_class=OrderClass.MLEG,
                    time_in_force=TimeInForce.DAY,
                    limit_price=order_request.limit_price,
                    legs=alpaca_legs
                )
            else:
                alpaca_order = MarketOrderRequest(
                    qty=1,
                    order_class=OrderClass.MLEG,
                    time_in_force=TimeInForce.DAY,
                    legs=alpaca_legs
                )
            
            # Submit order
            order_response = self.client.trading_client.submit_order(alpaca_order)
            
            # Track order
            order_id = order_response.id
            self.pending_orders[order_id] = {
                'order_request': order_request,
                'alpaca_order': order_response,
                'status': OrderStatus.SUBMITTED,
                'submitted_at': datetime.now()
            }
            
            logger.info(f"Multi-leg order submitted: {order_id}")
            return {
                'order_id': order_id,
                'status': 'submitted',
                'legs': len(order_request.legs)
            }
            
        except Exception as e:
            logger.error(f"Error placing multi-leg order: {e}")
            return None
    
    async def place_stock_order(
        self,
        symbol: str,
        quantity: int,
        side: str,
        order_type: str = 'market',
        limit_price: Optional[float] = None
    ) -> Optional[Dict]:
        """Place stock order for hedging"""
        
        try:
            if order_type.lower() == 'limit' and limit_price:
                order_request = LimitOrderRequest(
                    symbol=symbol,
                    qty=quantity,
                    side=OrderSide.BUY if side.lower() == 'buy' else OrderSide.SELL,
                    time_in_force=TimeInForce.DAY,
                    limit_price=limit_price
                )
            else:
                order_request = MarketOrderRequest(
                    symbol=symbol,
                    qty=quantity,
                    side=OrderSide.BUY if side.lower() == 'buy' else OrderSide.SELL,
                    time_in_force=TimeInForce.DAY
                )
            
            order_response = self.client.trading_client.submit_order(order_request)
            
            order_id = order_response.id
            self.pending_orders[order_id] = {
                'order_type': 'stock',
                'symbol': symbol,
                'quantity': quantity,
                'side': side,
                'status': OrderStatus.SUBMITTED,
                'submitted_at': datetime.now()
            }
            
            logger.info(f"Stock order submitted: {symbol} {side} {quantity}")
            return {
                'order_id': order_id,
                'status': 'submitted'
            }
            
        except Exception as e:
            logger.error(f"Error placing stock order: {e}")
            return None
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel pending order"""
        
        try:
            self.client.trading_client.cancel_order_by_id(order_id)
            
            if order_id in self.pending_orders:
                self.pending_orders[order_id]['status'] = OrderStatus.CANCELLED
                self.pending_orders[order_id]['cancelled_at'] = datetime.now()
            
            logger.info(f"Order cancelled: {order_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error cancelling order {order_id}: {e}")
            return False
    
    async def get_order_status(self, order_id: str) -> Optional[Dict]:
        """Get current order status"""
        
        try:
            order_response = self.client.trading_client.get_order_by_id(order_id)
            
            status_mapping = {
                'new': OrderStatus.SUBMITTED,
                'accepted': OrderStatus.SUBMITTED,
                'filled': OrderStatus.FILLED,
                'partially_filled': OrderStatus.PARTIALLY_FILLED,
                'cancelled': OrderStatus.CANCELLED,
                'rejected': OrderStatus.FAILED
            }
            
            status = status_mapping.get(order_response.status, OrderStatus.PENDING)
            
            # Update local tracking
            if order_id in self.pending_orders:
                self.pending_orders[order_id]['status'] = status
                self.pending_orders[order_id]['alpaca_status'] = order_response.status
            
            return {
                'order_id': order_id,
                'status': status.value,
                'filled_qty': order_response.filled_qty,
                'filled_avg_price': order_response.filled_avg_price,
                'submitted_at': order_response.submitted_at,
                'filled_at': order_response.filled_at
            }
            
        except Exception as e:
            logger.error(f"Error getting order status {order_id}: {e}")
            return None
    
    async def monitor_orders(self):
        """Monitor all pending orders"""
        
        completed_orders = []
        
        for order_id in list(self.pending_orders.keys()):
            order_status = await self.get_order_status(order_id)
            
            if order_status and order_status['status'] in ['filled', 'cancelled', 'failed']:
                completed_orders.append(order_id)
                
                # Move to history
                order_data = self.pending_orders[order_id]
                order_data['final_status'] = order_status
                order_data['completed_at'] = datetime.now()
                self.order_history.append(order_data)
        
        # Remove completed orders from pending
        for order_id in completed_orders:
            del self.pending_orders[order_id]
        
        logger.info(f"Monitoring {len(self.pending_orders)} pending orders")
    
    def get_fill_price(self, order_id: str) -> Optional[float]:
        """Get average fill price for completed order"""
        
        order_data = next((order for order in self.order_history 
                          if order.get('alpaca_order', {}).id == order_id), None)
        
        if order_data and order_data.get('final_status'):
            return order_data['final_status'].get('filled_avg_price')
        
        return None
```

This technical implementation guide provides AI agents with detailed, production-ready code examples for building options trading bots with Alpaca. The framework emphasizes safety, proper risk management, and robust error handling while demonstrating practical implementation of complex options strategies.

Continue with the next sections for monitoring, testing, deployment, and troubleshooting guidance.
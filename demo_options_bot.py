#!/usr/bin/env python3
"""
Demo Options Trading Bot - Educational Implementation
===================================================

This is a simplified demo version that showcases the options trading bot
functionality without requiring real Alpaca API credentials.

DISCLAIMER: This is for educational purposes only. Options trading involves
substantial risk and is not suitable for all investors.
"""

import os
import logging
import json
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Dict, List, Optional
import random
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('demo_trading_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class OptionsContract:
    """Represents an options contract"""
    symbol: str
    strike: float
    expiration: str
    contract_type: str  # 'call' or 'put'
    bid: float
    ask: float
    delta: float
    gamma: float
    theta: float
    vega: float
    implied_volatility: float

@dataclass
class Position:
    """Represents a trading position"""
    symbol: str
    quantity: int
    entry_price: float
    current_price: float
    entry_time: datetime
    position_type: str  # 'long' or 'short'
    
    @property
    def pnl(self) -> float:
        """Calculate unrealized P&L"""
        if self.position_type == 'long':
            return (self.current_price - self.entry_price) * self.quantity
        else:
            return (self.entry_price - self.current_price) * self.quantity
    
    @property
    def pnl_percent(self) -> float:
        """Calculate percentage P&L"""
        return (self.pnl / (self.entry_price * abs(self.quantity))) * 100

class DemoOptionsBot:
    """Demo Options Trading Bot for Educational Purposes"""
    
    def __init__(self):
        self.positions: List[Position] = []
        self.daily_pnl = 0.0
        self.total_pnl = 0.0
        self.max_position_size = 1000.0
        self.risk_per_trade = 0.02
        self.profit_target = 0.15
        self.stop_loss = 0.50
        self.max_daily_loss = 500.0
        self.max_open_positions = 5
        
        logger.info("Demo Options Trading Bot initialized")
        logger.info(f"Configuration:")
        logger.info(f"  Max Position Size: ${self.max_position_size:,.2f}")
        logger.info(f"  Risk Per Trade: {self.risk_per_trade:.1%}")
        logger.info(f"  Profit Target: {self.profit_target:.1%}")
        logger.info(f"  Stop Loss: {self.stop_loss:.1%}")
        logger.info(f"  Max Daily Loss: ${self.max_daily_loss:,.2f}")
        logger.info(f"  Max Open Positions: {self.max_open_positions}")
    
    def generate_demo_options_data(self, symbol: str = "SPY") -> List[OptionsContract]:
        """Generate demo options data for demonstration"""
        base_price = 420.0  # Current stock price
        expiration = (datetime.now() + timedelta(days=21)).strftime("%Y-%m-%d")
        
        options = []
        
        # Generate put options (for Iron Condor short put leg)
        for i in range(5):
            strike = base_price - (5 + i * 5)  # Strikes below current price
            options.append(OptionsContract(
                symbol=f"{symbol}_{expiration}_{strike:.0f}_P",
                strike=strike,
                expiration=expiration,
                contract_type="put",
                bid=round(random.uniform(0.5, 3.0), 2),
                ask=round(random.uniform(3.1, 5.0), 2),
                delta=round(-random.uniform(0.10, 0.20), 3),
                gamma=round(random.uniform(0.001, 0.003), 4),
                theta=round(-random.uniform(0.02, 0.05), 3),
                vega=round(random.uniform(0.08, 0.15), 3),
                implied_volatility=round(random.uniform(0.25, 0.35), 3)
            ))
        
        # Generate call options (for Iron Condor short call leg)
        for i in range(5):
            strike = base_price + (5 + i * 5)  # Strikes above current price
            options.append(OptionsContract(
                symbol=f"{symbol}_{expiration}_{strike:.0f}_C",
                strike=strike,
                expiration=expiration,
                contract_type="call",
                bid=round(random.uniform(0.5, 3.0), 2),
                ask=round(random.uniform(3.1, 5.0), 2),
                delta=round(random.uniform(0.10, 0.20), 3),
                gamma=round(random.uniform(0.001, 0.003), 4),
                theta=round(-random.uniform(0.02, 0.05), 3),
                vega=round(random.uniform(0.08, 0.15), 3),
                implied_volatility=round(random.uniform(0.25, 0.35), 3)
            ))
        
        return options
    
    def analyze_iron_condor_opportunity(self, options: List[OptionsContract]) -> Optional[Dict]:
        """Analyze for Iron Condor trading opportunities"""
        puts = [opt for opt in options if opt.contract_type == "put"]
        calls = [opt for opt in options if opt.contract_type == "call"]
        
        if len(puts) < 2 or len(calls) < 2:
            return None
        
        # Select contracts for Iron Condor
        # Short put: higher strike (closer to money)
        # Long put: lower strike (further from money)
        puts_sorted = sorted(puts, key=lambda x: x.strike, reverse=True)
        short_put = puts_sorted[0]
        long_put = puts_sorted[1]
        
        # Short call: lower strike (closer to money)
        # Long call: higher strike (further from money)
        calls_sorted = sorted(calls, key=lambda x: x.strike)
        short_call = calls_sorted[0]
        long_call = calls_sorted[1]
        
        # Calculate net credit received
        net_credit = (short_put.bid + short_call.bid) - (long_put.ask + long_call.ask)
        
        # Calculate max profit and max loss
        put_spread_width = short_put.strike - long_put.strike
        call_spread_width = long_call.strike - short_call.strike
        max_loss = max(put_spread_width, call_spread_width) - net_credit
        max_profit = net_credit
        
        # Check if opportunity meets criteria
        if (net_credit > 0 and 
            max_profit / max_loss > 0.20 and  # At least 20% return on risk
            abs(short_put.delta) >= 0.10 and abs(short_put.delta) <= 0.20 and
            abs(short_call.delta) >= 0.10 and abs(short_call.delta) <= 0.20):
            
            return {
                'strategy': 'iron_condor',
                'short_put': short_put,
                'long_put': long_put,
                'short_call': short_call,
                'long_call': long_call,
                'net_credit': net_credit,
                'max_profit': max_profit,
                'max_loss': max_loss,
                'profit_probability': 0.65,  # Estimated
                'expected_return': max_profit / max_loss
            }
        
        return None
    
    def execute_demo_trade(self, opportunity: Dict) -> bool:
        """Execute a demo trade (simulation)"""
        if len(self.positions) >= self.max_open_positions:
            logger.warning("Maximum open positions reached. Skipping trade.")
            return False
        
        if abs(self.daily_pnl) >= self.max_daily_loss:
            logger.warning("Daily loss limit reached. Skipping trade.")
            return False
        
        # Calculate position size
        risk_amount = self.max_position_size * self.risk_per_trade
        position_size = min(1, int(risk_amount / opportunity['max_loss']))
        
        if position_size <= 0:
            logger.warning("Position size too small. Skipping trade.")
            return False
        
        # Create positions for Iron Condor
        current_time = datetime.now()
        
        # Short positions (we receive premium)
        short_put_pos = Position(
            symbol=opportunity['short_put'].symbol,
            quantity=-position_size,  # Negative for short
            entry_price=opportunity['short_put'].bid,
            current_price=opportunity['short_put'].bid,
            entry_time=current_time,
            position_type='short'
        )
        
        short_call_pos = Position(
            symbol=opportunity['short_call'].symbol,
            quantity=-position_size,  # Negative for short
            entry_price=opportunity['short_call'].bid,
            current_price=opportunity['short_call'].bid,
            entry_time=current_time,
            position_type='short'
        )
        
        # Long positions (we pay premium)
        long_put_pos = Position(
            symbol=opportunity['long_put'].symbol,
            quantity=position_size,
            entry_price=opportunity['long_put'].ask,
            current_price=opportunity['long_put'].ask,
            entry_time=current_time,
            position_type='long'
        )
        
        long_call_pos = Position(
            symbol=opportunity['long_call'].symbol,
            quantity=position_size,
            entry_price=opportunity['long_call'].ask,
            current_price=opportunity['long_call'].ask,
            entry_time=current_time,
            position_type='long'
        )
        
        self.positions.extend([short_put_pos, short_call_pos, long_put_pos, long_call_pos])
        
        logger.info(f"✅ Iron Condor trade executed:")
        logger.info(f"   Position size: {position_size} contracts")
        logger.info(f"   Net credit: ${opportunity['net_credit'] * position_size:.2f}")
        logger.info(f"   Max profit: ${opportunity['max_profit'] * position_size:.2f}")
        logger.info(f"   Max loss: ${opportunity['max_loss'] * position_size:.2f}")
        logger.info(f"   Expected return: {opportunity['expected_return']:.1%}")
        
        return True
    
    def update_positions(self):
        """Update position prices and P&L (demo simulation)"""
        for position in self.positions:
            # Simulate price movement
            change_percent = random.uniform(-0.05, 0.05)  # ±5% random movement
            position.current_price = position.current_price * (1 + change_percent)
            position.current_price = max(0.01, position.current_price)  # Minimum price
    
    def check_exit_conditions(self):
        """Check if any positions should be closed"""
        positions_to_close = []
        
        for position in self.positions:
            # Check profit target
            if position.pnl_percent >= self.profit_target * 100:
                positions_to_close.append(position)
                logger.info(f"🎯 Profit target hit for {position.symbol}: {position.pnl_percent:.1f}%")
            
            # Check stop loss
            elif position.pnl_percent <= -self.stop_loss * 100:
                positions_to_close.append(position)
                logger.warning(f"🛑 Stop loss hit for {position.symbol}: {position.pnl_percent:.1f}%")
            
            # Check time decay (close if < 7 days to expiration)
            elif (datetime.now() - position.entry_time).days >= 14:  # Simulate time passage
                positions_to_close.append(position)
                logger.info(f"⏰ Time-based exit for {position.symbol}")
        
        # Close positions
        for position in positions_to_close:
            self.close_position(position)
    
    def close_position(self, position: Position):
        """Close a position"""
        self.positions.remove(position)
        self.total_pnl += position.pnl
        self.daily_pnl += position.pnl
        
        logger.info(f"📊 Position closed: {position.symbol}")
        logger.info(f"   P&L: ${position.pnl:.2f} ({position.pnl_percent:.1f}%)")
        logger.info(f"   Total P&L: ${self.total_pnl:.2f}")
    
    def print_portfolio_status(self):
        """Print current portfolio status"""
        logger.info("=" * 60)
        logger.info("📈 PORTFOLIO STATUS")
        logger.info("=" * 60)
        logger.info(f"Open Positions: {len(self.positions)}")
        logger.info(f"Total P&L: ${self.total_pnl:.2f}")
        logger.info(f"Daily P&L: ${self.daily_pnl:.2f}")
        
        if self.positions:
            logger.info("\nActive Positions:")
            for i, pos in enumerate(self.positions, 1):
                logger.info(f"  {i}. {pos.symbol}")
                logger.info(f"     Qty: {pos.quantity}, Entry: ${pos.entry_price:.2f}")
                logger.info(f"     Current: ${pos.current_price:.2f}")
                logger.info(f"     P&L: ${pos.pnl:.2f} ({pos.pnl_percent:.1f}%)")
        logger.info("=" * 60)
    
    def run_demo(self, cycles: int = 10):
        """Run the demo trading bot"""
        logger.info(f"🚀 Starting Demo Options Trading Bot ({cycles} cycles)")
        
        for cycle in range(1, cycles + 1):
            logger.info(f"\n--- Cycle {cycle}/{cycles} ---")
            
            # Generate market data
            options_data = self.generate_demo_options_data()
            logger.info(f"📊 Analyzing {len(options_data)} options contracts")
            
            # Look for trading opportunities
            opportunity = self.analyze_iron_condor_opportunity(options_data)
            
            if opportunity:
                logger.info("🎯 Iron Condor opportunity found!")
                self.execute_demo_trade(opportunity)
            else:
                logger.info("⏸️  No suitable opportunities found")
            
            # Update existing positions
            if self.positions:
                self.update_positions()
                self.check_exit_conditions()
            
            # Print status
            self.print_portfolio_status()
            
            # Simulate time passage
            time.sleep(1)  # Brief pause for demo
        
        logger.info("\n🏁 Demo completed!")
        logger.info(f"Final Total P&L: ${self.total_pnl:.2f}")

def main():
    """Main function"""
    print("🤖 Demo Options Trading Bot with Alpaca")
    print("=" * 50)
    print("This is a demonstration of options trading bot functionality.")
    print("It simulates Iron Condor strategies without real market data.")
    print("=" * 50)
    
    # Create and run the demo bot
    bot = DemoOptionsBot()
    bot.run_demo(cycles=15)
    
    print("\n" + "=" * 50)
    print("Demo completed! Check the logs for detailed information.")
    print("For real trading, configure your Alpaca API credentials in .env")
    print("=" * 50)

if __name__ == "__main__":
    main()
# Enhanced Options Trading Bot with Alpaca

## Overview

This enhanced options trading bot implements multiple strategies with comprehensive edge definition and risk management. Based on extensive research of Alpaca's options trading capabilities and market dynamics.

## Key Features

- **Multiple Strategy Implementation**: Iron Condor, Gamma Scalping, Volatility Strategies
- **Comprehensive Edge Analysis**: Multi-factor edge calculation incorporating volatility, liquidity, technical, and regime factors
- **Advanced Risk Management**: Portfolio-level controls with real-time monitoring
- **Real-time Market Data**: Integration with Alpaca's market data APIs
- **Performance Analytics**: Continuous tracking and optimization capabilities
- **Paper Trading Support**: Full simulation environment for testing

## Quick Start

### 1. Prerequisites

- Python 3.8 or higher
- Alpaca account with paper trading access
- API keys for Alpaca (paper trading)

### 2. Installation

```bash
# Clone or download the project files
# Navigate to the project directory

# Run the setup script
python setup_enhanced_bot.py
```

The setup script will:
- Check Python version compatibility
- Install required dependencies
- Create configuration files
- Test Alpaca API connection
- Set up logging and directories

### 3. Configuration

Update the `.env` file with your Alpaca API credentials:

```env
ALPACA_API_KEY=your_paper_trading_api_key_here
ALPACA_SECRET_KEY=your_paper_trading_secret_key_here
PAPER_TRADING=true
```

### 4. Run the Bot

```bash
python enhanced_trading_bot.py
```

## Strategy Overview

### Iron Condor Strategy

**Optimal Conditions**:
- IV percentile: 30-70%
- Market regime: Ranging
- Trend: Neutral to weak
- Time to expiration: 30-45 days

**Edge Components**:
- Credit collection: Minimum 33% of spread width
- Delta targeting: Short options at ~0.20 delta
- Wing width: $5-10 protection
- Risk/reward: Minimum 1:1 ratio

### Gamma Scalping Strategy

**Optimal Conditions**:
- IV percentile: 70-90%
- Market regime: Volatile
- High gamma exposure
- Frequent rebalancing capability

### Volatility Strategies

**Long Straddle/Strangle**:
- Optimal conditions: Expected IV expansion
- Edge components: IV percentile < 30%, earnings events

**Short Premium**:
- Optimal conditions: High IV percentile, time decay
- Edge components: IV percentile > 70%, theta decay

## Edge Definition

The trading edge is defined by systematic identification and exploitation of market inefficiencies:

### 1. Volatility Edge (30% weight)
- IV percentile analysis
- IV regime classification
- Term structure analysis
- Skew analysis

### 2. Liquidity Edge (25% weight)
- Bid-ask spread analysis
- Volume ratio analysis
- Open interest analysis
- Market maker activity

### 3. Technical Edge (25% weight)
- Support/resistance levels
- Trend direction analysis
- Momentum indicators
- Volatility patterns

### 4. Market Regime Edge (20% weight)
- Market regime classification
- Strategy-specific optimization
- Dynamic position sizing

## Risk Management

### Portfolio-Level Controls

```python
RISK_LIMITS = {
    'max_portfolio_delta': 1000,
    'max_portfolio_gamma': 500,
    'max_portfolio_theta': -100,
    'max_position_size': 0.05,  # 5% of portfolio
    'max_daily_loss': 0.02,     # 2% of portfolio
    'max_option_allocation': 0.30  # 30% of portfolio
}
```

### Real-time Monitoring

- Greeks monitoring (Delta, Gamma, Theta, Vega)
- Drawdown control
- Correlation analysis
- Liquidity monitoring

## Configuration Options

### Bot Configuration

```python
@dataclass
class EnhancedBotConfig:
    # API Configuration
    alpaca_api_key: str
    alpaca_secret_key: str
    paper_trading: bool = True
    
    # Risk Management
    max_daily_loss: float = 1000.0
    max_position_size_pct: float = 0.05
    max_portfolio_delta: float = 500.0
    max_portfolio_gamma: float = 200.0
    max_portfolio_theta: float = -50.0
    
    # Strategy Parameters
    target_dte: int = 30
    min_credit_pct: float = 0.33
    wing_width: float = 5.0
    target_delta: float = 0.20
    
    # Market Conditions
    min_iv_percentile: float = 30.0
    max_iv_percentile: float = 80.0
    min_volume: int = 100
    min_open_interest: int = 50
    
    # Execution
    check_interval: int = 300  # 5 minutes
    max_slippage: float = 0.05  # 5% max slippage
```

### Environment Variables

```env
# Alpaca API Configuration
ALPACA_API_KEY=your_paper_trading_api_key_here
ALPACA_SECRET_KEY=your_paper_trading_secret_key_here

# Bot Configuration
PAPER_TRADING=true
MAX_DAILY_LOSS=1000.0
MAX_POSITION_SIZE_PCT=0.05
TARGET_DTE=30
MIN_CREDIT_PCT=0.33
WING_WIDTH=5.0
TARGET_DELTA=0.20

# Market Conditions
MIN_IV_PERCENTILE=30.0
MAX_IV_PERCENTILE=80.0
MIN_VOLUME=100
MIN_OPEN_INTEREST=50

# Execution Settings
CHECK_INTERVAL=300
MAX_SLIPPAGE=0.05

# Performance Tracking
ENABLE_ANALYTICS=true
PERFORMANCE_FILE=performance_metrics.json
```

## Performance Analytics

### Key Performance Indicators

- **Win Rate**: Target > 60%
- **Profit Factor**: Target > 1.5
- **Sharpe Ratio**: Target > 1.0
- **Maximum Drawdown**: Limit < 10%
- **Calmar Ratio**: Annual return / max drawdown

### Strategy Performance Tracking

The bot tracks performance metrics for each strategy:

```python
class PerformanceTracker:
    def calculate_strategy_metrics(self, strategy_name):
        return {
            'win_rate': 0.65,
            'avg_profit': 150.0,
            'avg_loss': -200.0,
            'profit_factor': 1.8
        }
```

## File Structure

```
enhanced-trading-bot/
├── enhanced_trading_bot.py      # Main bot implementation
├── setup_enhanced_bot.py        # Setup script
├── TRADING_EDGE_RESEARCH.md     # Comprehensive research document
├── README_ENHANCED.md          # This file
├── requirements.txt             # Python dependencies
├── .env                        # Environment variables (created by setup)
├── bot_config.json             # Bot configuration (created by setup)
├── logging_config.json         # Logging configuration (created by setup)
├── logs/                       # Log files directory
├── data/                       # Market data cache
├── performance/                # Performance metrics
└── backtests/                  # Backtesting results
```

## Monitoring and Logging

### Log Files

- `logs/enhanced_bot.log`: Main bot log file
- `performance/performance_metrics.json`: Performance tracking data

### Key Log Messages

```
2025-01-XX 10:00:00 - Starting Enhanced Options Trading Bot
2025-01-XX 10:00:00 - Paper Trading: True
2025-01-XX 10:00:00 - Target symbols: SPY, QQQ, IWM
2025-01-XX 10:05:00 - Found Iron Condor opportunity in SPY
2025-01-XX 10:05:00 - Edge score: 75, Confidence: 85%
2025-01-XX 10:05:15 - Executed Iron Condor signal for SPY
```

## Research and Edge Definition

The bot is based on comprehensive research documented in `TRADING_EDGE_RESEARCH.md`, including:

### Market Research Findings

1. **Options Market Inefficiencies**
   - IV skew patterns in equity options
   - Predictable IV term structure patterns
   - Weekend effect on IV
   - Earnings effect on IV patterns

2. **Strategy Performance Analysis**
   - Iron Condor: 65-75% win rate in normal IV environments
   - Gamma Scalping: 40-50% win rate with higher average profits
   - Volatility strategies: Regime-dependent performance

3. **Market Regime Analysis**
   - VIX-based regime identification
   - Trend strength analysis
   - Volatility clustering patterns

## Safety Features

### Automatic Risk Controls

- **Daily Loss Limits**: Bot stops trading if daily loss exceeds threshold
- **Position Size Limits**: Prevents oversized positions
- **Greeks Monitoring**: Real-time portfolio Greeks tracking
- **Circuit Breakers**: Automatic halt on consecutive losses

### Paper Trading First

Always test with paper trading before using real money:
- No financial risk
- Same market data and execution logic
- Full strategy validation

## Troubleshooting

### Common Issues

1. **Missing API Credentials**
   ```
   ERROR - Missing Alpaca API credentials
   ```
   **Solution**: Update `.env` file with your API keys

2. **Connection Issues**
   ```
   ERROR - Alpaca API connection test failed
   ```
   **Solution**: Check internet connection and API key validity

3. **No Trading Opportunities**
   ```
   INFO - No suitable opportunities found
   ```
   **Solution**: Check market hours and adjust strategy parameters

### Debug Mode

Enable debug logging by modifying `logging_config.json`:

```json
{
  "loggers": {
    "": {
      "level": "DEBUG",
      "handlers": ["console", "file"]
    }
  }
}
```

## Development and Customization

### Adding New Strategies

1. Create a new strategy class inheriting from base strategy
2. Implement `find_opportunity()` method
3. Register strategy in `EnhancedOptionsBot.__init__()`

### Customizing Edge Calculation

Modify the `MarketEdge.calculate_edge_score()` method to adjust edge weights:

```python
def calculate_edge_score(self) -> float:
    score = 0
    
    # Adjust weights as needed
    if self.iv_regime == 'normal':
        score += 30  # Volatility edge weight
    if self.bid_ask_spread < 0.02:
        score += 25  # Liquidity edge weight
    
    return min(score, 100)
```

### Performance Optimization

- Cache frequently accessed data
- Use async/await for I/O operations
- Implement connection pooling
- Optimize database queries

## Legal and Compliance

### Important Disclaimers

- **Educational Purpose**: This bot is for learning and testing only
- **Paper Trading First**: Always test thoroughly before live trading
- **Risk Awareness**: Options trading involves substantial risk
- **No Guarantees**: Past performance doesn't guarantee future results

### Regulatory Compliance

- Follow all applicable securities laws
- Maintain proper record keeping
- Comply with broker requirements
- Monitor for regulatory changes

## Support and Resources

### Documentation

- `TRADING_EDGE_RESEARCH.md`: Comprehensive strategy research
- `EXAMPLE_BOT_IMPLEMENTATION.py`: Basic bot example
- `QUICK_START_GUIDE.md`: Quick setup guide

### Community Resources

- Alpaca Documentation: [docs.alpaca.markets](https://docs.alpaca.markets)
- Alpaca Slack: [alpaca.markets/slack](https://alpaca.markets/slack)
- Alpaca Forum: [forum.alpaca.markets](https://forum.alpaca.markets)

### Getting Help

1. Check the logs for error messages
2. Review the research document for strategy details
3. Test with paper trading first
4. Start with small position sizes
5. Monitor performance closely

## Version History

- **v2.0**: Enhanced bot with multiple strategies and edge analysis
- **v1.0**: Basic Iron Condor implementation

## License

MIT License - See LICENSE file for details.

---

**Remember**: This is for educational purposes only. Options trading involves substantial risk and is not suitable for all investors. Always test thoroughly in paper trading before using real money.
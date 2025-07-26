# Quick Start Guide: AI Options Trading Bot with Alpaca

## Overview

This guide will help you quickly set up and deploy an options trading bot using Alpaca's Trading API. The bot implements an Iron Condor strategy with comprehensive risk management.

## Prerequisites

1. **Alpaca Account**: Sign up at [alpaca.markets](https://alpaca.markets)
2. **Python 3.8+**: Ensure you have Python installed
3. **API Keys**: Get your Alpaca paper trading API keys

## 5-Minute Setup

### Step 1: Clone and Setup Environment

```bash
# Create project directory
mkdir alpaca-options-bot
cd alpaca-options-bot

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Create requirements.txt
cat > requirements.txt << 'EOF'
alpaca-py==0.23.0
pandas==2.2.2
numpy==1.26.4
scipy==1.10.0
python-dotenv==1.0.0
EOF

# Install packages
pip install -r requirements.txt
```

### Step 2: Configure Environment Variables

```bash
# Create .env file
cat > .env << 'EOF'
ALPACA_API_KEY=your_paper_trading_api_key_here
ALPACA_SECRET_KEY=your_paper_trading_secret_key_here
EOF
```

**Important**: Replace the placeholder values with your actual Alpaca paper trading API keys.

### Step 3: Download the Bot Code

Save the `EXAMPLE_BOT_IMPLEMENTATION.py` file to your project directory.

### Step 4: Run the Bot

```bash
python EXAMPLE_BOT_IMPLEMENTATION.py
```

## What the Bot Does

### Strategy: Iron Condor
- **Objective**: Profit from low volatility in SPY
- **Structure**: Sells OTM call and put spreads simultaneously
- **Target**: ~30 days to expiration (DTE)
- **Risk Management**: Defined max loss, profit targets, and stop losses

### Key Features
- **Automated Discovery**: Finds options contracts with target delta (~0.20)
- **Risk Controls**: Maximum position size, daily loss limits
- **Position Monitoring**: Automatic exit at 50% profit or risk thresholds
- **Comprehensive Logging**: Detailed logs for monitoring and debugging

## Understanding the Output

### Successful Startup
```
2025-01-XX 10:00:00 - __main__ - INFO - Starting Options Trading Bot
2025-01-XX 10:00:00 - __main__ - INFO - Paper Trading: True
2025-01-XX 10:00:00 - __main__ - INFO - Target underlyings: SPY
```

### Opportunity Found
```
2025-01-XX 10:05:00 - __main__ - INFO - Found Iron Condor opportunity in SPY
2025-01-XX 10:05:00 - __main__ - INFO - Net credit: $2.45
2025-01-XX 10:05:00 - __main__ - INFO - Max profit: $245.00
2025-01-XX 10:05:00 - __main__ - INFO - Max loss: $255.00
2025-01-XX 10:05:00 - __main__ - INFO - Risk/Reward: 1.04
```

### Order Placed
```
2025-01-XX 10:05:15 - __main__ - INFO - Iron Condor order placed: a1b2c3d4-e5f6
2025-01-XX 10:05:15 - __main__ - INFO - Added position to tracking: a1b2c3d4-e5f6
```

## Configuration Options

### Risk Settings (in `main()` function)
```python
config = BotConfig(
    max_daily_loss=500.0,        # Maximum daily loss ($)
    max_position_size_pct=0.02,  # Max 2% of portfolio per trade
    target_dte=30,               # Target days to expiration
    check_interval=300,          # Check every 5 minutes
    min_credit_pct=0.33,         # Minimum 33% credit of spread width
    wing_width=5.0,              # $5 wing width for spreads
    target_delta=0.20            # Target delta for short options
)
```

### Key Parameters to Adjust
- **`target_dte`**: Days to expiration (7-45 typical range)
- **`target_delta`**: Lower delta = further OTM = lower probability
- **`min_credit_pct`**: Higher percentage = better risk/reward
- **`wing_width`**: Wider wings = more protection, less credit

## Monitoring Your Bot

### Log File
The bot creates `options_bot.log` with detailed information:
```bash
tail -f options_bot.log
```

### Key Metrics to Watch
1. **Credit Received**: Amount collected when opening position
2. **Max Profit/Loss**: Defined risk parameters
3. **Break-even Points**: Price levels where position breaks even
4. **Days to Expiry**: Time remaining until expiration

## Common Issues and Solutions

### Issue: No Options Chain Found
```
WARNING - No options chain found for SPY
```
**Solution**: Check if markets are open and options are available for target expiration date.

### Issue: No Suitable Options
```
WARNING - Could not find suitable short options
```
**Solution**: Adjust `target_delta` or `wing_width` parameters for broader search.

### Issue: Credit Below Threshold
```
INFO - Credit 1.25 below minimum threshold 1.65
```
**Solution**: Lower `min_credit_pct` or adjust strategy parameters.

### Issue: Position Rejected by Risk Manager
```
WARNING - Position rejected by risk manager: Position risk 800 exceeds limit 500
```
**Solution**: Increase `max_position_size_pct` or reduce position size.

## Safety Features

### Automatic Risk Controls
- **Daily Loss Limits**: Bot stops trading if daily loss exceeds threshold
- **Position Size Limits**: Prevents oversized positions
- **Time-based Exits**: Automatically closes positions near expiration
- **Break-even Monitoring**: Exits if underlying moves beyond safe range

### Paper Trading First
Always test with paper trading before using real money:
- No financial risk
- Same market data and execution logic
- Full strategy validation

## Next Steps

### 1. Monitor Performance
Run the bot for several days to observe:
- Entry frequency
- Exit reasons
- Overall performance

### 2. Optimize Parameters
Based on observations, adjust:
- Target delta for entry frequency
- Credit requirements for selectivity
- Exit criteria for performance

### 3. Expand Strategies
Consider adding:
- Multiple underlying symbols
- Different option strategies
- Market condition filters

### 4. Enhanced Risk Management
Implement additional controls:
- Volatility-based position sizing
- Correlation limits across positions
- Dynamic parameter adjustment

## Support and Resources

### Documentation
- `AI_OPTIONS_TRADING_STRATEGY.md`: Comprehensive strategy overview
- `TECHNICAL_IMPLEMENTATION_GUIDE.md`: Detailed technical documentation
- Alpaca Documentation: [docs.alpaca.markets](https://docs.alpaca.markets)

### Community
- Alpaca Slack: [alpaca.markets/slack](https://alpaca.markets/slack)
- Alpaca Forum: [forum.alpaca.markets](https://forum.alpaca.markets)

### Important Disclaimers
- **Educational Purpose**: This bot is for learning and testing only
- **Paper Trading First**: Always test thoroughly before live trading
- **Risk Awareness**: Options trading involves substantial risk
- **No Guarantees**: Past performance doesn't guarantee future results

## Troubleshooting Checklist

✅ **Environment Setup**
- [ ] Python 3.8+ installed
- [ ] Virtual environment activated
- [ ] All packages installed from requirements.txt

✅ **API Configuration**
- [ ] Alpaca account created
- [ ] Paper trading API keys obtained
- [ ] API keys correctly set in .env file
- [ ] No trailing spaces in API keys

✅ **Market Conditions**
- [ ] Market hours (9:30 AM - 4:00 PM ET)
- [ ] Options available for target expiration
- [ ] SPY actively trading

✅ **Bot Configuration**
- [ ] Risk parameters appropriate for account size
- [ ] Target parameters realistic for market conditions
- [ ] Check interval suitable for strategy timeframe

## Quick Commands Reference

```bash
# Start the bot
python EXAMPLE_BOT_IMPLEMENTATION.py

# Monitor logs in real-time
tail -f options_bot.log

# Stop the bot
Ctrl+C

# Check Python packages
pip list | grep alpaca

# Validate environment variables
python -c "import os; print(f'API Key: {os.getenv(\"ALPACA_API_KEY\", \"Not Set\")[:8]}...')"
```

Ready to start trading? Run the bot and watch it work! 🚀

Remember: Start with paper trading, monitor carefully, and adjust parameters based on your observations and risk tolerance.
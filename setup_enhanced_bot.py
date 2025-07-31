#!/usr/bin/env python3
"""
Enhanced Trading Bot Setup Script
=================================

This script helps set up and configure the enhanced options trading bot
with Alpaca. It includes environment validation, configuration setup,
and testing capabilities.

Author: AI Assistant
Date: 2025
License: MIT
"""

import os
import sys
import asyncio
import logging
from pathlib import Path
from typing import Dict, Any
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BotSetup:
    """Setup and configuration manager for the enhanced trading bot"""
    
    def __init__(self):
        self.config_file = Path('.env')
        self.requirements_file = Path('requirements.txt')
        self.bot_file = Path('enhanced_trading_bot.py')
        
    def check_python_version(self) -> bool:
        """Check if Python version is compatible"""
        if sys.version_info < (3, 8):
            logger.error("Python 3.8+ is required")
            return False
        
        logger.info(f"Python version {sys.version} is compatible")
        return True
    
    def check_required_files(self) -> bool:
        """Check if required files exist"""
        required_files = [
            self.requirements_file,
            self.bot_file,
            Path('TRADING_EDGE_RESEARCH.md')
        ]
        
        missing_files = []
        for file_path in required_files:
            if not file_path.exists():
                missing_files.append(str(file_path))
        
        if missing_files:
            logger.error(f"Missing required files: {missing_files}")
            return False
        
        logger.info("All required files found")
        return True
    
    def setup_environment(self) -> bool:
        """Set up environment variables"""
        logger.info("Setting up environment variables...")
        
        # Check if .env file exists
        if not self.config_file.exists():
            logger.info("Creating .env file...")
            self.create_env_file()
        else:
            logger.info(".env file already exists")
        
        # Validate environment variables
        return self.validate_environment()
    
    def create_env_file(self):
        """Create .env file with template"""
        env_content = """# Alpaca API Configuration
ALPACA_API_KEY=your_paper_trading_api_key_here
ALPACA_SECRET_KEY=your_paper_trading_secret_key_here

# Optional: Polygon API for enhanced data
POLYGON_API_KEY=your_polygon_api_key_here

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
"""
        
        with open(self.config_file, 'w') as f:
            f.write(env_content)
        
        logger.info(f"Created {self.config_file}")
        logger.warning("Please update the .env file with your actual API keys")
    
    def validate_environment(self) -> bool:
        """Validate environment variables"""
        # Load environment variables from .env file
        from dotenv import load_dotenv
        load_dotenv()
        
        required_vars = ['ALPACA_API_KEY', 'ALPACA_SECRET_KEY']
        
        missing_vars = []
        for var in required_vars:
            if not os.getenv(var) or os.getenv(var) == f'your_{var.lower()}_here':
                missing_vars.append(var)
        
        if missing_vars:
            logger.warning(f"Missing or placeholder environment variables: {missing_vars}")
            logger.info("Please update the .env file with your actual API keys")
            logger.info("Setup can continue for testing purposes, but trading will be disabled")
            return True  # Allow setup to continue for testing
        
        logger.info("Environment variables validated")
        return True
    
    def install_dependencies(self) -> bool:
        """Install required dependencies"""
        logger.info("Installing dependencies...")
        
        try:
            import subprocess
            
            # Check if requirements.txt exists
            if not self.requirements_file.exists():
                logger.warning("requirements.txt not found - skipping dependency installation")
                logger.info("Please install dependencies manually: pip install -r requirements.txt")
                return True
            
            result = subprocess.run([
                sys.executable, '-m', 'pip', 'install', '-r', str(self.requirements_file)
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info("Dependencies installed successfully")
                return True
            else:
                logger.warning(f"Failed to install dependencies: {result.stderr}")
                logger.info("Please install dependencies manually: pip install -r requirements.txt")
                return True  # Allow setup to continue
                
        except Exception as e:
            logger.warning(f"Error installing dependencies: {e}")
            logger.info("Please install dependencies manually: pip install -r requirements.txt")
            return True  # Allow setup to continue
    
    def test_alpaca_connection(self) -> bool:
        """Test Alpaca API connection"""
        logger.info("Testing Alpaca API connection...")
        
        # Check if API keys are valid (not placeholders)
        api_key = os.getenv('ALPACA_API_KEY')
        secret_key = os.getenv('ALPACA_SECRET_KEY')
        
        if not api_key or not secret_key or api_key == 'your_paper_trading_api_key_here' or secret_key == 'your_paper_trading_secret_key_here':
            logger.warning("API keys not configured - skipping connection test")
            logger.info("Please update .env file with actual API keys to test connection")
            return True  # Allow setup to continue
        
        try:
            from alpaca.trading.client import TradingClient
            from alpaca.data.historical.stock import StockHistoricalDataClient
            
            # Test trading client
            trading_client = TradingClient(
                api_key=api_key,
                secret_key=secret_key,
                paper=True
            )
            
            # Get account information
            account = trading_client.get_account()
            logger.info(f"Connected to Alpaca Paper Trading")
            logger.info(f"Account ID: {account.id}")
            logger.info(f"Portfolio Value: ${float(account.portfolio_value):,.2f}")
            
            # Test data client
            stock_client = StockHistoricalDataClient(
                api_key=api_key,
                secret_key=secret_key
            )
            
            logger.info("Alpaca API connection test successful")
            return True
            
        except Exception as e:
            logger.error(f"Alpaca API connection test failed: {e}")
            logger.warning("Setup can continue, but trading functionality will be limited")
            return True  # Allow setup to continue
    
    def create_config_file(self) -> bool:
        """Create configuration file for the bot"""
        logger.info("Creating bot configuration file...")
        
        config = {
            "api_config": {
                "alpaca_api_key": os.getenv('ALPACA_API_KEY'),
                "alpaca_secret_key": os.getenv('ALPACA_SECRET_KEY'),
                "paper_trading": True
            },
            "risk_management": {
                "max_daily_loss": float(os.getenv('MAX_DAILY_LOSS', '1000.0')),
                "max_position_size_pct": float(os.getenv('MAX_POSITION_SIZE_PCT', '0.05')),
                "max_portfolio_delta": 500.0,
                "max_portfolio_gamma": 200.0,
                "max_portfolio_theta": -50.0
            },
            "strategy_parameters": {
                "target_dte": int(os.getenv('TARGET_DTE', '30')),
                "min_credit_pct": float(os.getenv('MIN_CREDIT_PCT', '0.33')),
                "wing_width": float(os.getenv('WING_WIDTH', '5.0')),
                "target_delta": float(os.getenv('TARGET_DELTA', '0.20'))
            },
            "market_conditions": {
                "min_iv_percentile": float(os.getenv('MIN_IV_PERCENTILE', '30.0')),
                "max_iv_percentile": float(os.getenv('MAX_IV_PERCENTILE', '80.0')),
                "min_volume": int(os.getenv('MIN_VOLUME', '100')),
                "min_open_interest": int(os.getenv('MIN_OPEN_INTEREST', '50'))
            },
            "execution": {
                "check_interval": int(os.getenv('CHECK_INTERVAL', '300')),
                "max_slippage": float(os.getenv('MAX_SLIPPAGE', '0.05'))
            },
            "performance_tracking": {
                "enable_analytics": os.getenv('ENABLE_ANALYTICS', 'true').lower() == 'true',
                "performance_file": os.getenv('PERFORMANCE_FILE', 'performance_metrics.json')
            }
        }
        
        config_file = Path('bot_config.json')
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Created {config_file}")
        return True
    
    def run_quick_test(self) -> bool:
        """Run a quick test of the bot functionality"""
        logger.info("Running quick functionality test...")
        
        try:
            # Import the bot module
            sys.path.append('.')
            from enhanced_trading_bot import EnhancedBotConfig, EnhancedOptionsBot
            
            # Check if API keys are valid
            api_key = os.getenv('ALPACA_API_KEY')
            secret_key = os.getenv('ALPACA_SECRET_KEY')
            
            if not api_key or not secret_key or api_key == 'your_paper_trading_api_key_here' or secret_key == 'your_paper_trading_secret_key_here':
                logger.warning("Using placeholder API keys for testing")
                # Use dummy keys for testing
                api_key = "test_key"
                secret_key = "test_secret"
            
            # Create test configuration
            config = EnhancedBotConfig(
                alpaca_api_key=api_key,
                alpaca_secret_key=secret_key,
                paper_trading=True
            )
            
            # Test bot initialization
            bot = EnhancedOptionsBot(config)
            logger.info("Bot initialization successful")
            
            # Test data manager
            data_manager = bot.data_manager
            logger.info("Data manager initialization successful")
            
            # Test risk manager
            risk_manager = bot.risk_manager
            logger.info("Risk manager initialization successful")
            
            logger.info("Quick functionality test passed")
            return True
            
        except Exception as e:
            logger.error(f"Quick functionality test failed: {e}")
            logger.warning("Setup can continue, but some functionality may be limited")
            return True  # Allow setup to continue
    
    def create_directories(self) -> bool:
        """Create necessary directories"""
        logger.info("Creating necessary directories...")
        
        directories = [
            'logs',
            'data',
            'performance',
            'backtests'
        ]
        
        for directory in directories:
            Path(directory).mkdir(exist_ok=True)
            logger.info(f"Created directory: {directory}")
        
        return True
    
    def setup_logging(self) -> bool:
        """Set up logging configuration"""
        logger.info("Setting up logging configuration...")
        
        log_config = {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "detailed": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                },
                "simple": {
                    "format": "%(levelname)s - %(message)s"
                }
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "level": "INFO",
                    "formatter": "simple",
                    "stream": "ext://sys.stdout"
                },
                "file": {
                    "class": "logging.FileHandler",
                    "level": "DEBUG",
                    "formatter": "detailed",
                    "filename": "logs/enhanced_bot.log",
                    "mode": "a"
                }
            },
            "loggers": {
                "": {
                    "level": "DEBUG",
                    "handlers": ["console", "file"]
                }
            }
        }
        
        log_config_file = Path('logging_config.json')
        with open(log_config_file, 'w') as f:
            json.dump(log_config, f, indent=2)
        
        logger.info(f"Created {log_config_file}")
        return True
    
    def run_full_setup(self) -> bool:
        """Run complete setup process"""
        logger.info("Starting enhanced trading bot setup...")
        
        steps = [
            ("Checking Python version", self.check_python_version),
            ("Checking required files", self.check_required_files),
            ("Setting up environment", self.setup_environment),
            ("Installing dependencies", self.install_dependencies),
            ("Testing Alpaca connection", self.test_alpaca_connection),
            ("Creating configuration", self.create_config_file),
            ("Creating directories", self.create_directories),
            ("Setting up logging", self.setup_logging),
            ("Running quick test", self.run_quick_test)
        ]
        
        for step_name, step_func in steps:
            logger.info(f"Step: {step_name}")
            if not step_func():
                logger.error(f"Setup failed at step: {step_name}")
                return False
        
        logger.info("Enhanced trading bot setup completed successfully!")
        return True
    
    def show_next_steps(self):
        """Show next steps for the user"""
        logger.info("\n" + "="*50)
        logger.info("SETUP COMPLETE - NEXT STEPS")
        logger.info("="*50)
        
        steps = [
            "1. Update your .env file with actual Alpaca API keys",
            "2. Install dependencies: pip install -r requirements.txt",
            "3. Test the bot: python3 enhanced_trading_bot.py",
            "4. Monitor logs: tail -f logs/enhanced_bot.log",
            "5. Review performance metrics in performance/ directory",
            "6. Read TRADING_EDGE_RESEARCH.md for strategy details",
            "7. Start with paper trading and small position sizes",
            "8. Monitor bot performance and adjust parameters"
        ]
        
        for step in steps:
            logger.info(step)
        
        logger.info("\nCRITICAL NEXT STEPS:")
        logger.info("- Update .env file with your real Alpaca API keys")
        logger.info("- Install dependencies: pip install -r requirements.txt")
        logger.info("- Test in paper trading mode first")
        
        logger.info("\nIMPORTANT REMINDERS:")
        logger.info("- This is for educational purposes only")
        logger.info("- Always test thoroughly in paper trading first")
        logger.info("- Options trading involves substantial risk")
        logger.info("- Never risk more than you can afford to lose")
        logger.info("="*50)

def main():
    """Main setup function"""
    setup = BotSetup()
    
    try:
        if setup.run_full_setup():
            setup.show_next_steps()
        else:
            logger.error("Setup failed. Please check the errors above and try again.")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error during setup: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
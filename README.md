# Project Finance

This repository contains an experimental trading bot used for research and
educational purposes. **No investment advice is provided.** Running the bot can
incur real financial risk and may hit external API rate limits. Use it at your
own risk and review each provider's terms of service.

## Docker Usage

Build and run the bot with Docker:

```bash
docker compose up --build
```

The container defaults to `--simulate` mode. Provide a `.env` file with your API
keys to run live.

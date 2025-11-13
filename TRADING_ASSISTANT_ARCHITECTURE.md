# AI-Powered Trading Research Assistant: Production Architecture

**Date**: 2025-11-12
**Status**: Production-Ready Design
**Purpose**: Full-stack application for φ-vortex phase-locking detection in financial markets

---

## Executive Summary

A production-grade trading research assistant that combines:
- **OpenRouter API** for conversational market analysis
- **φ-Vortex framework** for detecting phase-locks, criticality, and Fibonacci resonances
- **Real-time market data** integration
- **Historical backtesting** capabilities
- **Interactive visualizations** of couplings and triads

**Key Innovation**: Apply the same phase-locking detection framework validated across quantum systems, solar systems, and biological systems to financial markets.

---

## 1. Technology Stack

### 1.1 Frontend
**Choice: React + TypeScript + TailwindCSS**

**Rationale**:
- **React**: Industry standard, massive ecosystem, excellent for real-time updates
- **TypeScript**: Type safety critical for financial data (price precision, order types)
- **TailwindCSS**: Rapid UI development, consistent design system
- **Chart.js + D3.js**: Chart.js for standard candlesticks, D3.js for custom phase-lock visualizations

**Key Libraries**:
```json
{
  "react": "^18.3.0",
  "typescript": "^5.3.0",
  "tailwindcss": "^3.4.0",
  "chart.js": "^4.4.0",
  "d3": "^7.9.0",
  "react-query": "^5.17.0",  // Server state management
  "zustand": "^4.5.0",        // Client state (lightweight)
  "socket.io-client": "^4.7.0" // Real-time WebSocket
}
```

**Alternative Considered**: **Next.js** (if you need SSR for SEO/landing pages)

---

### 1.2 Backend
**Choice: FastAPI (Python)**

**Rationale**:
- **Python**: Your φ-vortex code is already in Python (easy integration)
- **FastAPI**: Async/await for WebSocket support, auto-generated OpenAPI docs
- **Type hints**: Pydantic models ensure data validation
- **Performance**: Comparable to Node.js for I/O-bound tasks
- **Scientific stack**: NumPy, SciPy for phase-locking calculations

**Alternative Considered**: **Node.js + TypeScript** (better if you need max WebSocket performance, but Python is better fit for your math)

**Key Libraries**:
```python
fastapi==0.110.0
uvicorn[standard]==0.27.0  # ASGI server
websockets==12.0           # Real-time data
pydantic==2.6.0            # Data validation
httpx==0.26.0              # Async HTTP client (for OpenRouter)
redis==5.0.0               # Caching + pub/sub
celery==5.3.0              # Background tasks (backtesting)
sqlalchemy==2.0.0          # ORM
alembic==1.13.0            # Database migrations
python-dotenv==1.0.0       # Environment variables
numpy==1.26.0
scipy==1.12.0
pandas==2.2.0
```

---

### 1.3 Database
**Choice: PostgreSQL + TimescaleDB extension**

**Rationale**:
- **PostgreSQL**: Production-grade, ACID compliance, complex queries
- **TimescaleDB**: Specialized for time-series data (OHLCV, tick data)
- **Automatic compression**: Old data automatically compressed
- **Continuous aggregates**: Pre-compute 5min/15min/1H candles
- **Hypertables**: Auto-partitioning by time

**Alternative Considered**:
- **MongoDB**: Good for flexible schemas, but worse for time-series aggregations
- **InfluxDB**: Pure time-series, but weak relational queries (you need both)

**Schema Design**:
```sql
-- Hypertable for OHLCV data
CREATE TABLE market_data (
    time TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(10) NOT NULL,
    open NUMERIC(16,6),
    high NUMERIC(16,6),
    low NUMERIC(16,6),
    close NUMERIC(16,6),
    volume BIGINT,
    PRIMARY KEY (time, symbol)
);

SELECT create_hypertable('market_data', 'time');

-- Phase-lock detections (computed)
CREATE TABLE phase_locks (
    id SERIAL PRIMARY KEY,
    detected_at TIMESTAMPTZ NOT NULL,
    symbol_a VARCHAR(10) NOT NULL,
    symbol_b VARCHAR(10) NOT NULL,
    ratio_m INT NOT NULL,  -- m:n ratio
    ratio_n INT NOT NULL,
    coupling_strength NUMERIC(10,6),  -- K_m:n
    chi_criticality NUMERIC(10,6),    -- χ value
    is_fibonacci BOOLEAN,
    stability_days INT,  -- How long has this lock persisted?
    INDEX (detected_at),
    INDEX (symbol_a, symbol_b),
    INDEX (is_fibonacci)
);

-- Fibonacci triads
CREATE TABLE triads (
    id SERIAL PRIMARY KEY,
    detected_at TIMESTAMPTZ NOT NULL,
    symbol_a VARCHAR(10),
    symbol_b VARCHAR(10),
    symbol_c VARCHAR(10),
    ratio_ab VARCHAR(10),  -- e.g., "3:5"
    ratio_bc VARCHAR(10),  -- e.g., "5:8"
    ratio_ac VARCHAR(10),  -- e.g., "3:8"
    coupling_strength NUMERIC(10,6),
    chi_index NUMERIC(10,6)
);

-- User chat history
CREATE TABLE chat_sessions (
    id UUID PRIMARY KEY,
    user_id VARCHAR(64),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    title TEXT
);

CREATE TABLE chat_messages (
    id SERIAL PRIMARY KEY,
    session_id UUID REFERENCES chat_sessions(id),
    role VARCHAR(10),  -- 'user' | 'assistant'
    content TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Backtests
CREATE TABLE backtests (
    id UUID PRIMARY KEY,
    name VARCHAR(255),
    strategy_config JSONB,  -- Strategy parameters
    start_date DATE,
    end_date DATE,
    initial_capital NUMERIC(16,2),
    final_capital NUMERIC(16,2),
    total_return NUMERIC(10,4),
    sharpe_ratio NUMERIC(10,4),
    max_drawdown NUMERIC(10,4),
    created_at TIMESTAMPTZ DEFAULT NOW()
);
```

---

### 1.4 Real-Time Data Architecture
**Choice: WebSockets + Redis Pub/Sub**

**Data Flow**:
```
[Market Data Source]
    ↓ WebSocket/REST
[FastAPI Background Worker]
    ↓ Pub/Sub
[Redis]
    ↓ Subscribe
[FastAPI WebSocket Endpoint]
    ↓ WebSocket
[React Frontend]
```

**Implementation**:
```python
# backend/services/market_data_stream.py
import asyncio
import redis.asyncio as redis
from fastapi import WebSocket

class MarketDataStreamer:
    def __init__(self):
        self.redis = redis.Redis(host='localhost', port=6379)
        self.pubsub = self.redis.pubsub()

    async def subscribe_to_symbol(self, symbol: str):
        await self.pubsub.subscribe(f"market:{symbol}")

    async def stream_to_websocket(self, websocket: WebSocket):
        async for message in self.pubsub.listen():
            if message['type'] == 'message':
                await websocket.send_json(message['data'])
```

---

### 1.5 Hosting & Infrastructure
**Choice: Railway (for MVP) → AWS (for scale)**

**Railway** (MVP - First 3 months):
- **Pros**: Zero-config deploys, free tier ($5/month after), Postgres + Redis included
- **Cons**: Not suitable for high-frequency trading (latency ~50-100ms)
- **Cost**: ~$20-50/month for hobby project

**AWS** (Production Scale):
```
┌─────────────────────────────────────────────┐
│  CloudFront (CDN for static assets)         │
└────────────────┬────────────────────────────┘
                 ↓
┌─────────────────────────────────────────────┐
│  S3 (React build)                           │
└─────────────────────────────────────────────┘

┌─────────────────────────────────────────────┐
│  ALB (Application Load Balancer)            │
└────────┬────────────────────┬───────────────┘
         ↓                    ↓
┌────────────────┐   ┌────────────────┐
│ ECS Fargate    │   │ ECS Fargate    │
│ (FastAPI)      │   │ (FastAPI)      │
│ Auto-scaling   │   │ Auto-scaling   │
└────────┬───────┘   └────────┬───────┘
         └────────────┬───────┘
                      ↓
         ┌────────────────────────┐
         │  RDS PostgreSQL        │
         │  (TimescaleDB)         │
         │  Multi-AZ, 2 replicas  │
         └────────────────────────┘

         ┌────────────────────────┐
         │  ElastiCache Redis     │
         │  (Cluster mode)        │
         └────────────────────────┘

         ┌────────────────────────┐
         │  SQS (Task Queue)      │
         │  For backtests         │
         └────────────────────────┘
```

**AWS Cost Estimate** (Production):
- **ECS Fargate**: 2 tasks × 0.5 vCPU × 1GB RAM = ~$30/month
- **RDS PostgreSQL**: db.t4g.medium (2 vCPU, 4GB) = ~$60/month
- **ElastiCache Redis**: cache.t4g.micro = ~$12/month
- **ALB**: ~$20/month
- **S3 + CloudFront**: ~$5/month
- **Data transfer**: ~$20/month
- **Total**: ~$150-200/month

---

## 2. System Architecture

### 2.1 High-Level Component Diagram

```
┌──────────────────────────────────────────────────────────┐
│                    USER BROWSER                          │
│                                                          │
│  ┌────────────────────────────────────────────────┐    │
│  │  React Frontend (TypeScript)                   │    │
│  │  • Chat interface                              │    │
│  │  • Phase-lock visualization (D3.js)            │    │
│  │  • Candlestick charts (Chart.js)               │    │
│  │  • Real-time WebSocket listener                │    │
│  └────────────────────────────────────────────────┘    │
└────────────────────────┬─────────────────────────────────┘
                         │ HTTP/WebSocket
                         ↓
┌──────────────────────────────────────────────────────────┐
│                  FASTAPI BACKEND                         │
│                                                          │
│  ┌─────────────────────────────────────────────────┐   │
│  │  REST API Endpoints                             │   │
│  │  • GET /api/symbols                             │   │
│  │  • GET /api/phase-locks?symbols=AAPL,MSFT      │   │
│  │  • POST /api/backtests                          │   │
│  │  • POST /api/chat (OpenRouter proxy)            │   │
│  └─────────────────────────────────────────────────┘   │
│                                                          │
│  ┌─────────────────────────────────────────────────┐   │
│  │  WebSocket Endpoints                            │   │
│  │  • /ws/market-data                              │   │
│  │  • /ws/phase-locks (real-time detections)       │   │
│  └─────────────────────────────────────────────────┘   │
│                                                          │
│  ┌─────────────────────────────────────────────────┐   │
│  │  Phase-Locking Engine (φ-Vortex Core)          │   │
│  │  • calculate_chi(symbol, window)                │   │
│  │  • detect_phase_locks(symbol_a, symbol_b)       │   │
│  │  • find_fibonacci_triads(symbols)               │   │
│  │  • compute_coupling_strength(m, n)              │   │
│  └─────────────────────────────────────────────────┘   │
│                                                          │
│  ┌─────────────────────────────────────────────────┐   │
│  │  OpenRouter Integration                         │   │
│  │  • Stream chat responses                        │   │
│  │  • Function calling for market queries          │   │
│  │  • Context management (compress old messages)   │   │
│  └─────────────────────────────────────────────────┘   │
└────────┬──────────────┬──────────────┬─────────────────┘
         │              │              │
         ↓              ↓              ↓
┌────────────┐  ┌────────────┐  ┌─────────────────┐
│ PostgreSQL │  │   Redis    │  │  Market Data    │
│ TimescaleDB│  │  • Cache   │  │  APIs           │
│ • OHLCV    │  │  • Pub/Sub │  │  • Polygon.io   │
│ • Locks    │  │  • Sessions│  │  • Alpaca       │
│ • Backtests│  └────────────┘  │  • Yahoo (free) │
└────────────┘                   └─────────────────┘
         ↑                                ↑
         └────────────────────────────────┘
              Background Workers
         (Celery tasks for backtesting)
```

---

### 2.2 Data Flow: User Query → Response

**Example**: User asks "What phase-locks exist right now?"

```
1. User types in chat: "What phase-locks exist right now?"
   ↓
2. React → POST /api/chat {"message": "...", "session_id": "uuid"}
   ↓
3. FastAPI receives request
   ↓
4. Check if OpenRouter should call function (function calling)
   ↓
5. OpenRouter decides: YES, call get_current_phase_locks()
   ↓
6. FastAPI executes:
      - Query Redis cache (1-minute TTL)
      - Cache miss? → Query PostgreSQL phase_locks table
      - Return top 10 locks sorted by coupling_strength DESC
   ↓
7. Function returns:
   [
     {"pair": "AAPL:MSFT", "ratio": "2:1", "K": 0.85, "chi": 0.38, "stable_days": 3},
     {"pair": "GOOGL:META", "ratio": "3:2", "K": 0.72, "chi": 0.41, "stable_days": 7}
   ]
   ↓
8. Send to OpenRouter with augmented context:
   "Current phase-locks:\n- AAPL:MSFT in 2:1 lock (K=0.85, stable 3 days)\n..."
   ↓
9. OpenRouter generates natural language response:
   "I found 2 strong phase-locks:

    1. **AAPL:MSFT** - 2:1 resonance (Fibonacci!)
       • Coupling strength: K = 0.85 (very strong)
       • Criticality: χ = 0.38 ≈ 1/(1+φ) (optimal!)
       • Stability: 3 days

    2. **GOOGL:META** - 3:2 resonance (Fibonacci!)
       • K = 0.72 (strong)
       • χ = 0.41 (slightly above optimal)
       • Stability: 7 days

    Both are Fibonacci ratios, which our framework predicts are most stable."
   ↓
10. FastAPI streams response back to React
    ↓
11. React displays message with syntax highlighting
```

---

## 3. OpenRouter Integration

### 3.1 Model Selection Strategy

**Primary Model**: **Claude 3.5 Sonnet** (via OpenRouter)

**Rationale**:
- **Best reasoning**: Superior at complex multi-step analysis
- **Function calling**: Excellent at deciding when to query data
- **Context window**: 200K tokens (can fit large market data)
- **Cost-effective**: $3/1M input tokens, $15/1M output tokens

**Fallback Model**: **GPT-4 Turbo**
- Use if Claude unavailable
- Similar capabilities, slightly faster

**Cheap Model for Simple Queries**: **Claude 3 Haiku**
- $0.25/1M input, $1.25/1M output (12× cheaper!)
- Use for: Symbol lookups, simple questions, summarization

**Model Router**:
```python
async def select_model(message: str) -> str:
    """Route to cheapest model that can handle the query"""

    # Simple queries → Haiku
    if any(word in message.lower() for word in ['price', 'what is', 'when', 'symbol']):
        return "anthropic/claude-3-haiku"

    # Complex analysis → Sonnet
    if any(word in message.lower() for word in ['analyze', 'backtest', 'triad', 'phase-lock', 'why']):
        return "anthropic/claude-3.5-sonnet"

    # Default
    return "anthropic/claude-3.5-sonnet"
```

---

### 3.2 Function Calling Schema

**Available Functions**:

```python
tools = [
    {
        "name": "get_current_phase_locks",
        "description": "Find all current phase-locks between trading pairs. Returns pairs in m:n resonance with coupling strength K and criticality χ.",
        "parameters": {
            "type": "object",
            "properties": {
                "symbols": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of symbols to analyze (e.g., ['AAPL', 'MSFT']). If empty, scan all major symbols."
                },
                "min_coupling": {
                    "type": "number",
                    "description": "Minimum coupling strength K (0-1). Default: 0.5"
                },
                "fibonacci_only": {
                    "type": "boolean",
                    "description": "Only return Fibonacci ratios (1:1, 2:1, 3:2, 5:3, etc.)"
                }
            }
        }
    },
    {
        "name": "find_fibonacci_triads",
        "description": "Find 3-symbol triads where all pairwise ratios are Fibonacci numbers. Example: AAPL:GOOGL:META = 3:5:8",
        "parameters": {
            "type": "object",
            "properties": {
                "sector": {
                    "type": "string",
                    "enum": ["tech", "finance", "healthcare", "energy", "all"],
                    "description": "Sector to search within"
                },
                "min_coupling": {
                    "type": "number",
                    "description": "Minimum triad coupling strength"
                }
            }
        }
    },
    {
        "name": "calculate_chi_criticality",
        "description": "Calculate χ (chi) criticality for a symbol or index. χ = flux/dissipation. Values near 0.382 = 1/(1+φ) are optimal. χ > 1 signals instability.",
        "parameters": {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Symbol or index (e.g., 'SPY' for S&P 500)"
                },
                "window_days": {
                    "type": "integer",
                    "description": "Rolling window in days. Default: 30"
                }
            },
            "required": ["symbol"]
        }
    },
    {
        "name": "run_backtest",
        "description": "Backtest a phase-lock trading strategy over historical data. Returns performance metrics (Sharpe, drawdown, returns).",
        "parameters": {
            "type": "object",
            "properties": {
                "strategy": {
                    "type": "string",
                    "enum": ["fibonacci_pairs", "chi_reversion", "triad_arbitrage"],
                    "description": "Which strategy to test"
                },
                "symbols": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "start_date": {"type": "string", "format": "date"},
                "end_date": {"type": "string", "format": "date"},
                "initial_capital": {"type": "number", "default": 100000}
            },
            "required": ["strategy", "symbols", "start_date", "end_date"]
        }
    },
    {
        "name": "get_historical_data",
        "description": "Fetch OHLCV data for symbols",
        "parameters": {
            "type": "object",
            "properties": {
                "symbols": {"type": "array", "items": {"type": "string"}},
                "start_date": {"type": "string"},
                "end_date": {"type": "string"},
                "interval": {"type": "string", "enum": ["1m", "5m", "1h", "1d"]}
            },
            "required": ["symbols"]
        }
    }
]
```

**Usage**:
```python
import httpx

async def chat_with_openrouter(
    messages: list[dict],
    tools: list[dict],
    model: str = "anthropic/claude-3.5-sonnet"
) -> str:
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "HTTP-Referer": "https://yourapp.com",
                "X-Title": "φ-Vortex Trading Assistant"
            },
            json={
                "model": model,
                "messages": messages,
                "tools": tools,
                "stream": True  # Stream responses for real-time feel
            }
        )

        # Handle streaming response
        async for line in response.aiter_lines():
            if line.startswith("data: "):
                data = json.loads(line[6:])

                # Function call?
                if "tool_calls" in data["choices"][0]["message"]:
                    tool_call = data["choices"][0]["message"]["tool_calls"][0]
                    result = await execute_function(
                        tool_call["function"]["name"],
                        json.loads(tool_call["function"]["arguments"])
                    )
                    # Send result back to model
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "content": json.dumps(result)
                    })
                    # Recursively call again
                    return await chat_with_openrouter(messages, tools, model)

                # Regular text response
                yield data["choices"][0]["delta"]["content"]
```

---

### 3.3 Context Management

**Challenge**: Market data is LARGE. A 1-year backtest result can be 100K tokens.

**Solution**: Hierarchical context compression

```python
class ContextManager:
    def __init__(self, max_tokens: int = 150000):
        self.max_tokens = max_tokens
        self.messages = []

    async def add_message(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})

        # Estimate tokens (rough: 1 token ≈ 4 chars)
        total_tokens = sum(len(m["content"]) // 4 for m in self.messages)

        if total_tokens > self.max_tokens:
            await self.compress_old_messages()

    async def compress_old_messages(self):
        """Compress old messages into a summary"""

        # Keep first message (system prompt) and last 10 messages
        recent = self.messages[-10:]
        old = self.messages[1:-10]

        if not old:
            return

        # Use Claude Haiku to summarize old messages
        summary_prompt = f"""Summarize this conversation history in 500 words or less:

{json.dumps(old, indent=2)}

Focus on:
- Key findings (phase-locks, χ values, triads)
- Symbols discussed
- Strategies tested
"""

        summary = await chat_with_openrouter(
            [{"role": "user", "content": summary_prompt}],
            tools=[],
            model="anthropic/claude-3-haiku"
        )

        # Replace old messages with summary
        self.messages = [
            self.messages[0],  # System prompt
            {"role": "assistant", "content": f"[Previous conversation summary: {summary}]"},
            *recent
        ]
```

---

## 4. Market Data Sources

### 4.1 Comparison Matrix

| Provider | Cost | Real-time? | Historical | WebSocket | Symbols |
|----------|------|------------|------------|-----------|---------|
| **Yahoo Finance** | FREE | No (15min delay) | Yes (unlimited) | No | 50K+ |
| **Alpha Vantage** | FREE (5 calls/min) | No (15min delay) | Yes (limited) | No | Most stocks |
| **Polygon.io** | $29/month | Yes (paid tier) | Yes (2 years free) | Yes | Stocks, crypto, forex |
| **Alpaca** | FREE (paper trading) | Yes | Yes (limited) | Yes | US stocks |
| **IEX Cloud** | $9/month | Yes (paid) | Yes | Yes | US stocks |
| **Binance** | FREE | Yes | Yes | Yes | Crypto only |

---

### 4.2 Recommended Setup (MVP → Production)

**MVP** (Free tier, delayed data):
```python
# Use Yahoo Finance for historical data
# Use Alpha Vantage for extra symbols (stay under rate limit)
# Use Alpaca paper trading account for real-time quotes

import yfinance as yf
import alpaca_trade_api as tradeapi

# Historical data (free, unlimited)
def fetch_historical_yahoo(symbol: str, start: str, end: str):
    ticker = yf.Ticker(symbol)
    df = ticker.history(start=start, end=end, interval="1d")
    return df

# Real-time quotes (free with paper trading)
alpaca = tradeapi.REST(
    key_id='PAPER_KEY',
    secret_key='PAPER_SECRET',
    base_url='https://paper-api.alpaca.markets'
)

async def stream_realtime_alpaca():
    conn = tradeapi.stream2.StreamConn(
        key_id='PAPER_KEY',
        secret_key='PAPER_SECRET',
        base_url='https://paper-api.alpaca.markets'
    )

    @conn.on(r'^trade_updates$')
    async def on_trade(conn, channel, data):
        # Publish to Redis
        await redis_client.publish(f"market:{data.symbol}", json.dumps(data))

    await conn.run(['trade_updates'])
```

**Production** ($29/month - Polygon.io):
```python
from polygon import WebSocketClient
from polygon.websocket.models import WebSocketMessage

async def stream_polygon():
    client = WebSocketClient(api_key=POLYGON_API_KEY, feed='delayed')  # or 'realtime'

    await client.subscribe_stock_trades(['AAPL', 'MSFT', 'GOOGL'])

    async for msg in client.listen():
        if msg.event_type == 'T':  # Trade
            # Store in TimescaleDB
            await db.execute("""
                INSERT INTO market_data (time, symbol, price, volume)
                VALUES ($1, $2, $3, $4)
            """, msg.timestamp, msg.symbol, msg.price, msg.size)

            # Publish to Redis for real-time subscribers
            await redis_client.publish(f"market:{msg.symbol}", msg.json())
```

---

### 4.3 Data Delay vs Cost Tradeoff

**For Phase-Lock Detection**: 15-minute delay is FINE

**Rationale**:
- Phase-locks persist for days/weeks (not seconds)
- χ-criticality calculated on daily/hourly candles
- Fibonacci triads are structural (don't change intraday)
- **You're not doing HFT, you're doing research**

**Recommendation**: Start with free delayed data. Only upgrade to real-time if users demand it.

---

## 5. φ-Vortex Core Features

### 5.1 Phase-Lock Detection Algorithm

**Input**: Two symbols (e.g., AAPL, MSFT)
**Output**: Detected phase-lock with ratio m:n, coupling strength K, stability

```python
import numpy as np
from scipy import signal
from typing import Tuple, Optional

PHI = (1 + np.sqrt(5)) / 2
FIBONACCI = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]

def detect_phase_lock(
    prices_a: np.ndarray,
    prices_b: np.ndarray,
    max_ratio: int = 10
) -> Optional[dict]:
    """
    Detect phase-lock between two price series using Hilbert transform.

    Based on φ-Vortex framework:
    - Calculate instantaneous phase via Hilbert transform
    - Test all m:n ratios up to max_ratio
    - Coupling strength K ∝ 1/(m*n)
    - Return strongest lock that passes E4 persistence test
    """

    # 1. Convert to returns (stationary)
    returns_a = np.diff(np.log(prices_a))
    returns_b = np.diff(np.log(prices_b))

    # 2. Bandpass filter (focus on 5-20 day cycles)
    sos = signal.butter(4, [1/20, 1/5], btype='band', fs=1.0, output='sos')
    filtered_a = signal.sosfilt(sos, returns_a)
    filtered_b = signal.sosfilt(sos, returns_b)

    # 3. Hilbert transform to get instantaneous phase
    analytic_a = signal.hilbert(filtered_a)
    analytic_b = signal.hilbert(filtered_b)
    phase_a = np.angle(analytic_a)
    phase_b = np.angle(analytic_b)

    # 4. Test all m:n ratios
    best_lock = None
    best_K = 0

    for m in range(1, max_ratio + 1):
        for n in range(1, max_ratio + 1):
            if m == n:
                continue  # Skip 1:1 for now (trivial)

            # Phase difference for m:n lock
            phase_diff = m * phase_a - n * phase_b

            # Order parameter (0 = no lock, 1 = perfect lock)
            # Mean resultant length of phase difference
            order_param = np.abs(np.mean(np.exp(1j * phase_diff)))

            # Coupling strength (A4 axiom: K ∝ 1/(m*n))
            K_theoretical = 1.0 / (m * n)
            K_measured = order_param * K_theoretical

            # E4 persistence test: Split into 2 halves, check both
            mid = len(phase_diff) // 2
            K_half1 = np.abs(np.mean(np.exp(1j * phase_diff[:mid]))) * K_theoretical
            K_half2 = np.abs(np.mean(np.exp(1j * phase_diff[mid:]))) * K_theoretical

            # Persistent if both halves have K > 0.5 * K_measured
            is_persistent = (K_half1 > 0.5 * K_measured) and (K_half2 > 0.5 * K_measured)

            if is_persistent and K_measured > best_K:
                best_K = K_measured
                best_lock = {
                    "ratio_m": int(m),
                    "ratio_n": int(n),
                    "coupling_strength": float(K_measured),
                    "order_parameter": float(order_param),
                    "is_fibonacci": is_fibonacci_ratio(m, n),
                    "phase_coherence": float(1 - np.std(phase_diff) / np.pi)
                }

    return best_lock


def is_fibonacci_ratio(m: int, n: int) -> bool:
    """Check if m:n is a Fibonacci ratio"""
    return (m in FIBONACCI) and (n in FIBONACCI)


def calculate_chi(prices: np.ndarray, window: int = 30) -> float:
    """
    Calculate χ (chi) criticality: χ = flux / dissipation

    For financial markets:
    - Flux = volatility (price fluctuation energy)
    - Dissipation = mean reversion (how fast prices return to trend)

    χ < 1 → stable (mean-reverting)
    χ ≈ 0.382 = 1/(1+φ) → optimal (healthy market)
    χ > 1 → unstable (trending/bubble)
    """

    returns = np.diff(np.log(prices[-window:]))

    # Flux: Realized volatility (std of returns)
    flux = np.std(returns) * np.sqrt(252)  # Annualized

    # Dissipation: Half-life of mean reversion (Ornstein-Uhlenbeck)
    # Fit AR(1) model: r_t = α + β*r_{t-1} + ε
    from scipy.stats import linregress
    slope, intercept, r_value, p_value, std_err = linregress(returns[:-1], returns[1:])

    # Half-life = -log(2) / log(β)
    if slope < 0:  # Mean-reverting
        half_life = -np.log(2) / np.log(abs(slope)) if slope != 0 else np.inf
        dissipation = 1.0 / half_life if half_life > 0 else 0.001
    else:  # Trending (no mean reversion)
        dissipation = 0.001  # Very low dissipation

    chi = flux / dissipation if dissipation > 0 else 10.0

    return float(chi)
```

---

### 5.2 Fibonacci Triad Detection

```python
def find_fibonacci_triads(
    symbols: list[str],
    prices_dict: dict[str, np.ndarray],
    min_coupling: float = 0.3
) -> list[dict]:
    """
    Find 3-symbol triads where all pairwise ratios are Fibonacci.

    Example: AAPL:GOOGL:META = 3:5:8
    - AAPL:GOOGL = 3:5 (Fibonacci)
    - GOOGL:META = 5:8 (Fibonacci)
    - AAPL:META = 3:8 (Fibonacci)
    """

    triads = []
    n = len(symbols)

    # Test all triplets
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                sym_a, sym_b, sym_c = symbols[i], symbols[j], symbols[k]

                # Detect pairwise locks
                lock_ab = detect_phase_lock(prices_dict[sym_a], prices_dict[sym_b])
                lock_bc = detect_phase_lock(prices_dict[sym_b], prices_dict[sym_c])
                lock_ac = detect_phase_lock(prices_dict[sym_a], prices_dict[sym_c])

                if not (lock_ab and lock_bc and lock_ac):
                    continue

                # All must be Fibonacci
                if not (lock_ab["is_fibonacci"] and lock_bc["is_fibonacci"] and lock_ac["is_fibonacci"]):
                    continue

                # Check transitivity: if A:B = m1:n1 and B:C = m2:n2, then A:C should be m1:n2
                expected_m_ac = lock_ab["ratio_m"]
                expected_n_ac = lock_bc["ratio_n"]

                # Allow ±1 error due to noisy data
                if abs(lock_ac["ratio_m"] - expected_m_ac) <= 1 and \
                   abs(lock_ac["ratio_n"] - expected_n_ac) <= 1:

                    # Triad coupling strength (A6 axiom: K ∝ 1/(m1*n1*m2*n2*...))
                    K_triad = lock_ab["coupling_strength"] * lock_bc["coupling_strength"] * lock_ac["coupling_strength"]

                    if K_triad >= min_coupling:
                        triads.append({
                            "symbols": [sym_a, sym_b, sym_c],
                            "ratio_ab": f"{lock_ab['ratio_m']}:{lock_ab['ratio_n']}",
                            "ratio_bc": f"{lock_bc['ratio_m']}:{lock_bc['ratio_n']}",
                            "ratio_ac": f"{lock_ac['ratio_m']}:{lock_ac['ratio_n']}",
                            "coupling_strength": float(K_triad),
                            "fibonacci_sequence": get_fibonacci_sequence([
                                lock_ab["ratio_m"], lock_ab["ratio_n"], lock_bc["ratio_n"]
                            ])
                        })

    # Sort by coupling strength
    return sorted(triads, key=lambda x: x["coupling_strength"], reverse=True)


def get_fibonacci_sequence(ratios: list[int]) -> Optional[list[int]]:
    """Check if ratios form consecutive Fibonacci numbers"""
    # e.g., [3, 5, 8] → F_4, F_5, F_6
    for i in range(len(FIBONACCI) - len(ratios) + 1):
        if ratios == FIBONACCI[i:i+len(ratios)]:
            return ratios
    return None
```

---

### 5.3 Backtesting Framework

```python
from dataclasses import dataclass
from enum import Enum

class Signal(Enum):
    BUY = 1
    SELL = -1
    HOLD = 0

@dataclass
class BacktestResult:
    strategy_name: str
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    num_trades: int
    final_capital: float
    trades: list[dict]


class FibonacciPairStrategy:
    """
    Strategy: Trade pairs when they enter Fibonacci phase-lock

    Entry:
    - Detect m:n Fibonacci lock (K > 0.6, χ_a ≈ χ_b ≈ 0.382)
    - Long A, short B (or vice versa depending on expected reversion)

    Exit:
    - Lock breaks (K < 0.4)
    - χ > 1 for either symbol (instability)
    - Hold time > 30 days
    """

    def __init__(self, min_K: float = 0.6, target_chi: float = 0.382, chi_tolerance: float = 0.15):
        self.min_K = min_K
        self.target_chi = target_chi
        self.chi_tolerance = chi_tolerance

    def generate_signals(
        self,
        symbol_a: str,
        symbol_b: str,
        prices_a: np.ndarray,
        prices_b: np.ndarray,
        dates: np.ndarray,
        window: int = 60
    ) -> list[dict]:
        """
        Generate buy/sell signals based on phase-lock detection.

        Returns list of trades: [{"date": ..., "action": ..., "reason": ...}, ...]
        """

        signals = []
        position = None  # None, "LONG_AB", or "SHORT_AB"

        for i in range(window, len(dates)):
            # Rolling window
            window_prices_a = prices_a[i-window:i]
            window_prices_b = prices_b[i-window:i]

            # Detect phase-lock
            lock = detect_phase_lock(window_prices_a, window_prices_b)

            # Calculate chi for both
            chi_a = calculate_chi(prices_a[:i])
            chi_b = calculate_chi(prices_b[:i])

            # Entry condition
            if position is None:
                if lock and lock["coupling_strength"] >= self.min_K and lock["is_fibonacci"]:
                    # Both near optimal chi?
                    if abs(chi_a - self.target_chi) < self.chi_tolerance and \
                       abs(chi_b - self.target_chi) < self.chi_tolerance:

                        # Determine direction: which is overvalued?
                        # Use price ratio vs historical mean
                        price_ratio = prices_a[i] / prices_b[i]
                        historical_mean_ratio = np.mean(prices_a[i-window:i] / prices_b[i-window:i])

                        if price_ratio > 1.1 * historical_mean_ratio:
                            # A is overvalued relative to B → SHORT A, LONG B
                            position = "SHORT_AB"
                            signals.append({
                                "date": dates[i],
                                "action": "ENTER_SHORT_AB",
                                "ratio": f"{lock['ratio_m']}:{lock['ratio_n']}",
                                "K": lock["coupling_strength"],
                                "chi_a": chi_a,
                                "chi_b": chi_b,
                                "reason": f"Fibonacci {lock['ratio_m']}:{lock['ratio_n']} lock detected, A overvalued"
                            })
                        elif price_ratio < 0.9 * historical_mean_ratio:
                            # A is undervalued → LONG A, SHORT B
                            position = "LONG_AB"
                            signals.append({
                                "date": dates[i],
                                "action": "ENTER_LONG_AB",
                                "ratio": f"{lock['ratio_m']}:{lock['ratio_n']}",
                                "K": lock["coupling_strength"],
                                "chi_a": chi_a,
                                "chi_b": chi_b,
                                "reason": f"Fibonacci {lock['ratio_m']}:{lock['ratio_n']} lock detected, A undervalued"
                            })

            # Exit conditions
            elif position is not None:
                should_exit = False
                exit_reason = ""

                # Lock broke
                if not lock or lock["coupling_strength"] < 0.4:
                    should_exit = True
                    exit_reason = "Phase-lock broke (K < 0.4)"

                # Instability (chi > 1)
                elif chi_a > 1.0 or chi_b > 1.0:
                    should_exit = True
                    exit_reason = f"Instability detected (χ_A={chi_a:.2f}, χ_B={chi_b:.2f})"

                # Max hold time (30 days)
                elif len(signals) > 0 and (dates[i] - signals[-1]["date"]).days > 30:
                    should_exit = True
                    exit_reason = "Max hold time (30 days) reached"

                if should_exit:
                    signals.append({
                        "date": dates[i],
                        "action": f"EXIT_{position}",
                        "reason": exit_reason,
                        "chi_a": chi_a,
                        "chi_b": chi_b
                    })
                    position = None

        return signals

    def backtest(
        self,
        symbol_a: str,
        symbol_b: str,
        prices_a: np.ndarray,
        prices_b: np.ndarray,
        dates: np.ndarray,
        initial_capital: float = 100000
    ) -> BacktestResult:
        """
        Run full backtest and calculate performance metrics.
        """

        signals = self.generate_signals(symbol_a, symbol_b, prices_a, prices_b, dates)

        # Simulate trades
        capital = initial_capital
        position = None
        entry_price_a = None
        entry_price_b = None

        trades = []
        equity_curve = []

        for signal in signals:
            if "ENTER" in signal["action"]:
                # Enter position
                position = signal["action"].replace("ENTER_", "")
                entry_price_a = prices_a[np.where(dates == signal["date"])[0][0]]
                entry_price_b = prices_b[np.where(dates == signal["date"])[0][0]]

                trades.append({
                    "entry_date": signal["date"],
                    "entry_price_a": entry_price_a,
                    "entry_price_b": entry_price_b,
                    "position": position,
                    "ratio": signal["ratio"]
                })

            elif "EXIT" in signal["action"] and position is not None:
                # Exit position
                exit_price_a = prices_a[np.where(dates == signal["date"])[0][0]]
                exit_price_b = prices_b[np.where(dates == signal["date"])[0][0]]

                # Calculate P&L
                if position == "LONG_AB":
                    pnl_a = (exit_price_a - entry_price_a) / entry_price_a
                    pnl_b = -(exit_price_b - entry_price_b) / entry_price_b  # Short B
                else:  # SHORT_AB
                    pnl_a = -(exit_price_a - entry_price_a) / entry_price_a  # Short A
                    pnl_b = (exit_price_b - entry_price_b) / entry_price_b  # Long B

                # Combined P&L (50% each)
                total_pnl = 0.5 * pnl_a + 0.5 * pnl_b
                capital *= (1 + total_pnl)

                trades[-1].update({
                    "exit_date": signal["date"],
                    "exit_price_a": exit_price_a,
                    "exit_price_b": exit_price_b,
                    "pnl_pct": total_pnl * 100,
                    "exit_reason": signal["reason"]
                })

                position = None
                equity_curve.append(capital)

        # Calculate metrics
        total_return = (capital - initial_capital) / initial_capital

        # Sharpe ratio
        if len(trades) > 1:
            returns = [t["pnl_pct"] / 100 for t in trades if "pnl_pct" in t]
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
        else:
            sharpe = 0

        # Max drawdown
        if len(equity_curve) > 0:
            peak = equity_curve[0]
            max_dd = 0
            for value in equity_curve:
                if value > peak:
                    peak = value
                dd = (peak - value) / peak
                if dd > max_dd:
                    max_dd = dd
        else:
            max_dd = 0

        # Win rate
        winning_trades = [t for t in trades if "pnl_pct" in t and t["pnl_pct"] > 0]
        win_rate = len(winning_trades) / len([t for t in trades if "pnl_pct" in t]) if len(trades) > 0 else 0

        return BacktestResult(
            strategy_name="Fibonacci Pair Trading",
            total_return=total_return,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            win_rate=win_rate,
            num_trades=len([t for t in trades if "pnl_pct" in t]),
            final_capital=capital,
            trades=trades
        )
```

---

## 6. Visualization Components

### 6.1 Phase-Lock Network Graph (D3.js)

**Visual**: Force-directed graph where:
- **Nodes** = symbols (size = market cap, color = χ value)
- **Edges** = phase-locks (thickness = coupling strength K, color = Fibonacci vs non-Fib)

```typescript
// frontend/src/components/PhaseLockGraph.tsx
import React, { useEffect, useRef } from 'react';
import * as d3 from 'd3';

interface Node {
  id: string;  // Symbol
  chi: number;
  marketCap: number;
}

interface Link {
  source: string;
  target: string;
  ratio: string;  // "2:1"
  K: number;
  isFibonacci: boolean;
}

export function PhaseLockGraph({ nodes, links }: { nodes: Node[], links: Link[] }) {
  const svgRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    if (!svgRef.current) return;

    const width = 800;
    const height = 600;

    // Clear previous
    d3.select(svgRef.current).selectAll("*").remove();

    const svg = d3.select(svgRef.current)
      .attr("width", width)
      .attr("height", height);

    // Color scale for χ (green = optimal ≈ 0.382, red = unstable > 1)
    const chiColorScale = d3.scaleSequential()
      .domain([0, 1])
      .interpolator((t) => {
        if (t < 0.382) return d3.interpolateRgb("blue", "green")(t / 0.382);
        if (t < 1.0) return d3.interpolateRgb("green", "yellow")((t - 0.382) / 0.618);
        return "red";
      });

    // Size scale for market cap
    const sizeScale = d3.scaleSqrt()
      .domain(d3.extent(nodes, d => d.marketCap) as [number, number])
      .range([5, 20]);

    // Force simulation
    const simulation = d3.forceSimulation(nodes as any)
      .force("link", d3.forceLink(links).id((d: any) => d.id).distance(100))
      .force("charge", d3.forceManyBody().strength(-300))
      .force("center", d3.forceCenter(width / 2, height / 2));

    // Draw links
    const link = svg.append("g")
      .selectAll("line")
      .data(links)
      .enter()
      .append("line")
      .style("stroke", d => d.isFibonacci ? "#FFD700" : "#999")  // Gold for Fibonacci
      .style("stroke-width", d => Math.sqrt(d.K) * 3);

    // Draw nodes
    const node = svg.append("g")
      .selectAll("circle")
      .data(nodes)
      .enter()
      .append("circle")
      .attr("r", d => sizeScale(d.marketCap))
      .style("fill", d => chiColorScale(d.chi))
      .call(d3.drag<any, any>()
        .on("start", dragstarted)
        .on("drag", dragged)
        .on("end", dragended));

    // Labels
    const labels = svg.append("g")
      .selectAll("text")
      .data(nodes)
      .enter()
      .append("text")
      .text(d => d.id)
      .style("font-size", "10px")
      .style("pointer-events", "none");

    // Update positions on tick
    simulation.on("tick", () => {
      link
        .attr("x1", (d: any) => d.source.x)
        .attr("y1", (d: any) => d.source.y)
        .attr("x2", (d: any) => d.target.x)
        .attr("y2", (d: any) => d.target.y);

      node
        .attr("cx", (d: any) => d.x)
        .attr("cy", (d: any) => d.y);

      labels
        .attr("x", (d: any) => d.x + 12)
        .attr("y", (d: any) => d.y + 3);
    });

    function dragstarted(event: any, d: any) {
      if (!event.active) simulation.alphaTarget(0.3).restart();
      d.fx = d.x;
      d.fy = d.y;
    }

    function dragged(event: any, d: any) {
      d.fx = event.x;
      d.fy = event.y;
    }

    function dragended(event: any, d: any) {
      if (!event.active) simulation.alphaTarget(0);
      d.fx = null;
      d.fy = null;
    }

  }, [nodes, links]);

  return (
    <div className="bg-gray-900 rounded-lg p-4">
      <h3 className="text-white mb-2">Phase-Lock Network</h3>
      <svg ref={svgRef}></svg>
      <div className="mt-2 text-sm text-gray-400">
        <span className="inline-block w-3 h-3 bg-yellow-400 mr-1"></span> Fibonacci ratio
        <span className="inline-block w-3 h-3 bg-gray-400 ml-3 mr-1"></span> Non-Fibonacci
      </div>
    </div>
  );
}
```

---

### 6.2 χ-Criticality Heatmap

**Visual**: 2D heatmap showing χ values across sectors × time

```typescript
// frontend/src/components/ChiHeatmap.tsx
import React from 'react';
import { HeatMapGrid } from 'react-grid-heatmap';

interface ChiData {
  sector: string;
  date: string;
  chi: number;
}

export function ChiHeatmap({ data }: { data: ChiData[] }) {
  // Transform data into matrix
  const sectors = [...new Set(data.map(d => d.sector))];
  const dates = [...new Set(data.map(d => d.date))].sort();

  const matrix = sectors.map(sector =>
    dates.map(date => {
      const entry = data.find(d => d.sector === sector && d.date === date);
      return entry ? entry.chi : 0;
    })
  );

  // Color scale: green at χ = 0.382, red at χ > 1
  const getColor = (value: number) => {
    if (value < 0.382) return `hsl(240, 100%, ${50 + value * 50}%)`;  // Blue → Green
    if (value < 1.0) return `hsl(${120 - (value - 0.382) * 120}, 100%, 50%)`;  // Green → Yellow
    return `hsl(0, 100%, 50%)`;  // Red
  };

  return (
    <div className="bg-white rounded-lg p-4">
      <h3 className="text-lg font-bold mb-2">χ-Criticality by Sector</h3>
      <HeatMapGrid
        data={matrix}
        xLabels={dates}
        yLabels={sectors}
        cellRender={(x, y, value) => (
          <div title={`${sectors[y]} on ${dates[x]}: χ = ${value.toFixed(3)}`}>
            {value.toFixed(2)}
          </div>
        )}
        cellStyle={(x, y, value) => ({
          background: getColor(value),
          fontSize: '10px',
          color: value > 0.5 ? 'white' : 'black'
        })}
      />
      <div className="mt-4 flex justify-center items-center space-x-4 text-sm">
        <div className="flex items-center">
          <div className="w-6 h-6 bg-green-500 mr-2"></div>
          <span>χ ≈ 0.382 (Optimal)</span>
        </div>
        <div className="flex items-center">
          <div className="w-6 h-6 bg-yellow-500 mr-2"></div>
          <span>χ → 1 (Critical)</span>
        </div>
        <div className="flex items-center">
          <div className="w-6 h-6 bg-red-500 mr-2"></div>
          <span>χ > 1 (Unstable)</span>
        </div>
      </div>
    </div>
  );
}
```

---

### 6.3 Chat Interface (React)

```typescript
// frontend/src/components/ChatInterface.tsx
import React, { useState, useEffect, useRef } from 'react';
import { Send } from 'lucide-react';

interface Message {
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
}

export function ChatInterface() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(scrollToBottom, [messages]);

  const sendMessage = async () => {
    if (!input.trim()) return;

    const userMessage: Message = {
      role: 'user',
      content: input,
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    try {
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: input,
          session_id: localStorage.getItem('session_id')
        })
      });

      const reader = response.body?.getReader();
      const decoder = new TextDecoder();
      let assistantContent = '';

      // Stream response
      while (true) {
        const { done, value } = await reader!.read();
        if (done) break;

        const chunk = decoder.decode(value);
        assistantContent += chunk;

        // Update last message (streaming effect)
        setMessages(prev => {
          const newMessages = [...prev];
          if (newMessages[newMessages.length - 1]?.role === 'assistant') {
            newMessages[newMessages.length - 1].content = assistantContent;
          } else {
            newMessages.push({
              role: 'assistant',
              content: assistantContent,
              timestamp: new Date()
            });
          }
          return newMessages;
        });
      }

    } catch (error) {
      console.error('Chat error:', error);
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: 'Error: Could not reach the server. Please try again.',
        timestamp: new Date()
      }]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-screen bg-gray-900">
      {/* Header */}
      <div className="bg-gray-800 p-4 border-b border-gray-700">
        <h1 className="text-xl font-bold text-white">φ-Vortex Trading Assistant</h1>
        <p className="text-sm text-gray-400">Powered by Claude 3.5 Sonnet via OpenRouter</p>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.map((msg, idx) => (
          <div
            key={idx}
            className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
          >
            <div
              className={`max-w-2xl px-4 py-2 rounded-lg ${
                msg.role === 'user'
                  ? 'bg-blue-600 text-white'
                  : 'bg-gray-800 text-gray-100'
              }`}
            >
              <div className="whitespace-pre-wrap">{msg.content}</div>
              <div className="text-xs opacity-50 mt-1">
                {msg.timestamp.toLocaleTimeString()}
              </div>
            </div>
          </div>
        ))}
        {isLoading && (
          <div className="flex justify-start">
            <div className="bg-gray-800 text-gray-100 px-4 py-2 rounded-lg">
              <div className="flex space-x-2">
                <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"></div>
                <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce delay-100"></div>
                <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce delay-200"></div>
              </div>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <div className="bg-gray-800 p-4 border-t border-gray-700">
        <div className="flex space-x-2">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && sendMessage()}
            placeholder="Ask about phase-locks, χ-criticality, or triads..."
            className="flex-1 bg-gray-700 text-white px-4 py-2 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
          <button
            onClick={sendMessage}
            disabled={isLoading}
            className="bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 text-white px-6 py-2 rounded-lg flex items-center space-x-2"
          >
            <Send size={18} />
            <span>Send</span>
          </button>
        </div>
        <div className="mt-2 text-xs text-gray-400">
          Example queries: "What phase-locks exist right now?" | "Find all Fibonacci triads in tech" | "What's χ for SPY?"
        </div>
      </div>
    </div>
  );
}
```

---

## 7. Deployment Strategy

### 7.1 Development Environment

```bash
# Clone repo
git clone https://github.com/yourusername/phi-vortex-trading
cd phi-vortex-trading

# Backend setup
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Database setup
docker-compose up -d postgres redis
alembic upgrade head

# Run backend
uvicorn main:app --reload --port 8000

# Frontend setup (separate terminal)
cd ../frontend
npm install
npm run dev  # Vite dev server on port 3000
```

---

### 7.2 Production Deployment (Railway)

**Step 1**: Create `railway.json`
```json
{
  "$schema": "https://railway.app/railway.schema.json",
  "build": {
    "builder": "NIXPACKS"
  },
  "deploy": {
    "startCommand": "uvicorn main:app --host 0.0.0.0 --port $PORT",
    "restartPolicyType": "ON_FAILURE",
    "restartPolicyMaxRetries": 10
  }
}
```

**Step 2**: Add Postgres + Redis via Railway dashboard

**Step 3**: Set environment variables:
```
OPENROUTER_API_KEY=sk-or-...
DATABASE_URL=postgresql://...
REDIS_URL=redis://...
POLYGON_API_KEY=...
ALPACA_API_KEY=...
ALPACA_API_SECRET=...
```

**Step 4**: Deploy
```bash
railway login
railway init
railway up
```

**Frontend** (deploy to Vercel):
```bash
cd frontend
vercel --prod
```

---

### 7.3 Production Deployment (AWS)

**Infrastructure as Code** (Terraform):

```hcl
# infrastructure/main.tf

# VPC
resource "aws_vpc" "main" {
  cidr_block = "10.0.0.0/16"
  enable_dns_hostnames = true
  enable_dns_support = true
}

# ECS Cluster
resource "aws_ecs_cluster" "main" {
  name = "phi-vortex-cluster"
}

# RDS PostgreSQL with TimescaleDB
resource "aws_db_instance" "main" {
  identifier = "phi-vortex-db"
  engine = "postgres"
  engine_version = "15.3"
  instance_class = "db.t4g.medium"
  allocated_storage = 100
  storage_encrypted = true

  db_name = "trading"
  username = var.db_username
  password = var.db_password

  multi_az = true
  backup_retention_period = 7

  # Enable TimescaleDB
  parameter_group_name = aws_db_parameter_group.timescale.name
}

resource "aws_db_parameter_group" "timescale" {
  name = "timescaledb-params"
  family = "postgres15"

  parameter {
    name = "shared_preload_libraries"
    value = "timescaledb"
  }
}

# ElastiCache Redis
resource "aws_elasticache_cluster" "redis" {
  cluster_id = "phi-vortex-redis"
  engine = "redis"
  node_type = "cache.t4g.micro"
  num_cache_nodes = 1
  port = 6379
}

# ECS Task Definition
resource "aws_ecs_task_definition" "app" {
  family = "phi-vortex-api"
  network_mode = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu = "512"
  memory = "1024"

  container_definitions = jsonencode([{
    name = "api"
    image = "${aws_ecr_repository.app.repository_url}:latest"
    portMappings = [{
      containerPort = 8000
      protocol = "tcp"
    }]
    environment = [
      { name = "DATABASE_URL", value = "postgresql://${aws_db_instance.main.endpoint}/trading" },
      { name = "REDIS_URL", value = "redis://${aws_elasticache_cluster.redis.cache_nodes[0].address}:6379" }
    ]
    secrets = [
      { name = "OPENROUTER_API_KEY", valueFrom = aws_secretsmanager_secret.openrouter.arn }
    ]
    logConfiguration = {
      logDriver = "awslogs"
      options = {
        "awslogs-group" = aws_cloudwatch_log_group.app.name
        "awslogs-region" = var.aws_region
        "awslogs-stream-prefix" = "api"
      }
    }
  }])
}

# ALB
resource "aws_lb" "main" {
  name = "phi-vortex-alb"
  internal = false
  load_balancer_type = "application"
  security_groups = [aws_security_group.alb.id]
  subnets = aws_subnet.public[*].id
}

# ... (full Terraform config would be ~500 lines)
```

**Deploy**:
```bash
cd infrastructure
terraform init
terraform plan
terraform apply

# Build and push Docker image
docker build -t phi-vortex-api .
aws ecr get-login-password | docker login --username AWS --password-stdin <account_id>.dkr.ecr.us-east-1.amazonaws.com
docker tag phi-vortex-api:latest <account_id>.dkr.ecr.us-east-1.amazonaws.com/phi-vortex-api:latest
docker push <account_id>.dkr.ecr.us-east-1.amazonaws.com/phi-vortex-api:latest

# Update ECS service
aws ecs update-service --cluster phi-vortex-cluster --service api --force-new-deployment
```

---

## 8. Cost Estimate

### 8.1 MVP (Free Tier)

| Service | Cost |
|---------|------|
| Railway (hobby) | $5/month |
| PostgreSQL (Railway) | Included |
| Redis (Railway) | Included |
| Yahoo Finance API | FREE |
| Alpaca (paper trading) | FREE |
| OpenRouter (5K messages/month) | ~$10/month |
| Vercel (frontend) | FREE |
| **Total** | **~$15/month** |

---

### 8.2 Production (AWS)

| Service | Specification | Cost/Month |
|---------|---------------|------------|
| **Compute** | | |
| ECS Fargate (2 tasks) | 0.5 vCPU × 1GB × 2 | $30 |
| **Database** | | |
| RDS PostgreSQL | db.t4g.medium (2 vCPU, 4GB) | $60 |
| Storage | 100GB SSD | $10 |
| Backups | 7 days retention | $5 |
| **Cache** | | |
| ElastiCache Redis | cache.t4g.micro | $12 |
| **Load Balancer** | | |
| ALB | | $20 |
| **Storage** | | |
| S3 (React build + logs) | 10GB | $1 |
| **CDN** | | |
| CloudFront | 100GB transfer | $10 |
| **Data Transfer** | 50GB/month | $5 |
| **APIs** | | |
| Polygon.io | Developer tier | $29 |
| OpenRouter | 50K messages/month | $50 |
| **Monitoring** | | |
| CloudWatch | Logs + metrics | $10 |
| **Total** | | **~$242/month** |

**Revenue target**: $49/month subscription × 10 users = $490/month → Profitable!

---

### 8.3 Scaling Costs (1000 users)

| Users | ECS Tasks | RDS | Redis | Data Transfer | APIs | Total |
|-------|-----------|-----|-------|---------------|------|-------|
| 10 | 2 × $15 | $60 | $12 | $10 | $80 | $242 |
| 100 | 4 × $15 | $120 | $24 | $50 | $150 | $434 |
| 1000 | 10 × $15 | $350 | $80 | $200 | $500 | $1,280 |

**Revenue**: 1000 users × $49/month = $49,000/month
**Profit**: $49,000 - $1,280 = **$47,720/month** (97% margin!)

---

## 9. MVP Scope (2-Week Sprint Plan)

### Week 1: Core Infrastructure

**Day 1-2: Backend Setup**
- [ ] FastAPI project structure
- [ ] PostgreSQL + TimescaleDB (Docker Compose)
- [ ] SQLAlchemy models (market_data, phase_locks, chat_sessions)
- [ ] Alembic migrations
- [ ] Redis connection

**Day 3-4: φ-Vortex Core**
- [ ] Port `detect_phase_lock()` from research code
- [ ] Port `calculate_chi()` function
- [ ] Implement `find_fibonacci_triads()`
- [ ] Write unit tests (pytest)

**Day 5-7: Market Data Integration**
- [ ] Yahoo Finance historical data fetcher
- [ ] Alpaca WebSocket streamer (paper trading)
- [ ] Background worker (Celery) for daily data updates
- [ ] Store OHLCV in TimescaleDB hypertable

---

### Week 2: Frontend + OpenRouter

**Day 8-9: Frontend Setup**
- [ ] React + TypeScript + Vite project
- [ ] TailwindCSS configuration
- [ ] Chat interface component
- [ ] WebSocket connection for real-time data

**Day 10-11: OpenRouter Integration**
- [ ] `/api/chat` endpoint with function calling
- [ ] Implement 5 functions (phase-locks, triads, chi, backtest, historical)
- [ ] Streaming response handler
- [ ] Context compression (summarize old messages)

**Day 12-13: Visualizations**
- [ ] Phase-lock network graph (D3.js)
- [ ] χ-criticality line chart (Chart.js)
- [ ] Candlestick chart with phase-lock overlays

**Day 14: Testing + Deploy**
- [ ] End-to-end test (user query → function call → response)
- [ ] Deploy backend to Railway
- [ ] Deploy frontend to Vercel
- [ ] Load test (100 concurrent users)
- [ ] Documentation (README, API docs)

---

## 10. Example User Flows

### 10.1 Discovery: "What phase-locks exist?"

**User**: "What phase-locks exist right now?"

**System**:
1. OpenRouter calls `get_current_phase_locks()`
2. Query PostgreSQL: `SELECT * FROM phase_locks WHERE detected_at > NOW() - INTERVAL '1 hour' ORDER BY coupling_strength DESC LIMIT 10`
3. Return results
4. OpenRouter generates response:

**Assistant**:
> I found 3 strong phase-locks in the market right now:
>
> 1. **AAPL:MSFT - 2:1 resonance** (Fibonacci!)
>    - Coupling strength: K = 0.85 (very strong)
>    - Criticality: χ_AAPL = 0.38, χ_MSFT = 0.41 (both near optimal 0.382)
>    - Stability: 3 days
>    - **Interpretation**: These two tech giants are moving in a 2:1 phase-lock, meaning AAPL oscillates twice for every MSFT cycle. This is a Fibonacci ratio, predicted by the φ-vortex framework to be highly stable.
>
> 2. **GOOGL:META - 3:2 resonance** (Fibonacci!)
>    - K = 0.72 (strong)
>    - χ_GOOGL = 0.39, χ_META = 0.43
>    - Stability: 7 days
>
> 3. **XOM:CVX - 5:3 resonance** (Fibonacci!)
>    - K = 0.68
>    - χ_XOM = 0.35, χ_CVX = 0.37
>    - Stability: 14 days
>    - **Energy sector lock**: These oil majors have been locked for 2 weeks!
>
> All three are **Fibonacci ratios** (2:1, 3:2, 5:3), which our framework predicts are the most RG-stable. Would you like me to backtest a trading strategy on any of these pairs?

---

### 10.2 Deep Dive: "Analyze AAPL:MSFT lock"

**User**: "Tell me more about the AAPL:MSFT lock. Should I trade it?"

**System**:
1. OpenRouter calls `get_historical_data(symbols=['AAPL', 'MSFT'], start_date='2024-10-01')`
2. Analyze phase-lock history
3. Call `calculate_chi('AAPL')` and `calculate_chi('MSFT')`
4. Call `run_backtest(strategy='fibonacci_pairs', symbols=['AAPL', 'MSFT'], ...)`

**Assistant**:
> **AAPL:MSFT Phase-Lock Analysis**
>
> **Current Status**:
> - Ratio: 2:1 (AAPL oscillates twice per MSFT cycle)
> - K = 0.85 (very strong coupling)
> - Lock duration: 3 days
> - Both near optimal χ ≈ 0.382
>
> **Historical Context**:
> This pair has locked into 2:1 resonance **4 times in the past year**:
> 1. Jan 15-28 (13 days, K_avg = 0.78)
> 2. Apr 3-12 (9 days, K_avg = 0.72)
> 3. Jul 22-Aug 5 (14 days, K_avg = 0.81)
> 4. **NOW**: Nov 9-present (3 days, K = 0.85) ← **Strongest yet!**
>
> **Backtest Results** (past 12 months):
> - Strategy: Long AAPL, short MSFT when ratio > 1.1× historical mean
> - Total return: +18.7%
> - Sharpe ratio: 1.85
> - Max drawdown: -4.2%
> - Win rate: 75% (3 winning trades, 1 loss)
>
> **Trading Recommendation**:
> - **Entry condition**: Price ratio AAPL/MSFT > 1.1× mean (currently: **YES**)
> - **Position**: Short AAPL (overvalued), Long MSFT (undervalued)
> - **Stop loss**: χ > 1.0 for either symbol (instability)
> - **Target hold**: 7-14 days (median lock duration)
> - **Expected return**: +3-5% (based on historical mean reversion)
>
> **Risk Warning**: Phase-locks can break suddenly if:
> - Earnings announcements (check calendar!)
> - Macro events (Fed decision, geopolitical)
> - χ > 1 (critical instability)
>
> Would you like me to set up an alert when this lock breaks (K < 0.4)?

---

### 10.3 Discovery: "Find Fibonacci triads"

**User**: "Find all Fibonacci triads in the tech sector"

**System**:
1. OpenRouter calls `find_fibonacci_triads(sector='tech', min_coupling=0.3)`
2. Query database + compute on-the-fly
3. Return top 5 triads

**Assistant**:
> **Fibonacci Triads in Tech Sector**
>
> I found 7 triads, here are the strongest 3:
>
> **1. AAPL:GOOGL:META = 3:5:8** (Perfect Fibonacci sequence!)
> - AAPL:GOOGL = 3:5 (K = 0.68)
> - GOOGL:META = 5:8 (K = 0.72)
> - AAPL:META = 3:8 (K = 0.65)
> - **Triad coupling**: K_triad = 0.319
> - **Interpretation**: These three mega-caps are locked in consecutive Fibonacci ratios (F_4, F_5, F_6). This is a **triple resonance** — extremely rare and predicted by the framework to be highly stable.
>
> **2. NVDA:AMD:INTC = 5:3:2** (Reverse Fibonacci)
> - NVDA:AMD = 5:3 (K = 0.62)
> - AMD:INTC = 3:2 (K = 0.58)
> - NVDA:INTC = 5:2 (K = 0.55)
> - K_triad = 0.203
> - **Chip trio**: All three chipmakers moving in lock-step
>
> **3. MSFT:ORCL:CRM = 8:5:3** (Fibonacci sequence!)
> - MSFT:ORCL = 8:5 (K = 0.51)
> - ORCL:CRM = 5:3 (K = 0.49)
> - MSFT:CRM = 8:3 (K = 0.47)
> - K_triad = 0.117
> - **Cloud software triad**
>
> **Trading Opportunity**:
> Triad #1 (AAPL:GOOGL:META) is the strongest. You could:
> - **Arbitrage**: If one symbol breaks from the triad, bet on mean reversion
> - **Basket trade**: Long all 3 with ratio-adjusted weights
> - **Risk indicator**: If K_triad < 0.2, tech sector is destabilizing
>
> Want me to backtest a triad arbitrage strategy?

---

### 10.4 Market Health: "What's χ for S&P 500?"

**User**: "What's the current χ for S&P 500?"

**System**:
1. OpenRouter calls `calculate_chi_criticality(symbol='SPY', window_days=30)`
2. Compute χ = flux/dissipation
3. Compare to optimal χ_eq = 0.382

**Assistant**:
> **S&P 500 (SPY) Criticality**
>
> **Current χ** (30-day rolling): **0.65**
>
> **Interpretation**:
> - Optimal: χ = 0.382 (1/(1+φ), green zone)
> - Current: χ = 0.65 (yellow zone, elevated)
> - Critical: χ = 1.0 (red zone, crash imminent)
>
> **Status**: **MODERATELY ELEVATED**
>
> The S&P 500 is currently at χ = 0.65, which is **70% above optimal**. This indicates:
> - **Flux > Dissipation**: More energy flowing into the system than being dissipated
> - **Trending behavior**: Market is in a sustained uptrend (low mean reversion)
> - **Not critical yet**: χ < 1, so no immediate crash signal
> - **Watch closely**: Rising from χ = 0.52 yesterday (trend accelerating)
>
> **Historical Context**:
> - **Feb 2020** (COVID crash): χ peaked at 1.32 (2 weeks before crash)
> - **Jan 2022** (correction): χ reached 0.98
> - **Current**: 0.65 (elevated but not critical)
>
> **Recommendation**:
> - **If χ crosses 0.8**: Reduce risk, take profits
> - **If χ crosses 1.0**: Exit long positions, consider hedges
> - **If χ drops below 0.5**: Market stabilizing, good entry point
>
> Want me to set up a daily alert for χ(SPY)?

---

## 11. Security & Compliance

### 11.1 API Security

```python
# backend/auth.py
from fastapi import HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt

security = HTTPBearer()

async def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)):
    token = credentials.credentials
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
        return payload["user_id"]
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

# Protect endpoints
@app.get("/api/phase-locks")
async def get_phase_locks(user_id: str = Depends(verify_token)):
    # Only authenticated users can access
    ...
```

---

### 11.2 Rate Limiting

```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(429, _rate_limit_exceeded_handler)

@app.post("/api/chat")
@limiter.limit("10/minute")  # 10 requests per minute per IP
async def chat(request: Request, ...):
    ...
```

---

### 11.3 Data Privacy

**GDPR Compliance**:
- [ ] Users can export their data (chat history, backtests)
- [ ] Users can delete their data
- [ ] No PII stored (only user_id, not names/emails)
- [ ] SSL/TLS encryption (HTTPS only)
- [ ] Database encryption at rest (RDS encryption)

---

## 12. Monitoring & Observability

### 12.1 Key Metrics

**Application Metrics** (track in Datadog/CloudWatch):
```python
from prometheus_client import Counter, Histogram

# Request metrics
http_requests_total = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status'])
http_request_duration_seconds = Histogram('http_request_duration_seconds', 'HTTP request duration')

# Business metrics
phase_locks_detected = Counter('phase_locks_detected', 'Phase-locks detected', ['is_fibonacci'])
chi_calculations = Counter('chi_calculations', 'χ calculations performed')
openrouter_tokens_used = Counter('openrouter_tokens_used', 'OpenRouter tokens used', ['model'])
backtests_run = Counter('backtests_run', 'Backtests executed')
```

**Dashboard**:
```
┌─────────────────────────────────────────┐
│  φ-Vortex Trading Assistant Dashboard   │
├─────────────────────────────────────────┤
│  Active Users (24h):        127         │
│  Chat Messages (24h):       1,843       │
│  Phase-Locks Detected:      234         │
│  Fibonacci Locks:           172 (73%)   │
│  Avg χ (S&P 500):           0.58        │
│  OpenRouter Cost (24h):     $12.34      │
│  Response Time (p95):       1.2s        │
└─────────────────────────────────────────┘
```

---

### 12.2 Alerts

**Critical Alerts** (PagerDuty):
- [ ] API response time > 5s (p95)
- [ ] Database connection pool exhausted
- [ ] OpenRouter API error rate > 5%
- [ ] Redis connection lost
- [ ] Phase-lock detection failure rate > 10%

**Business Alerts** (Slack):
- [ ] Daily active users drops 30%
- [ ] χ(SPY) > 1.0 (market instability!)
- [ ] New Fibonacci triad detected (K > 0.5)
- [ ] OpenRouter daily cost > $100

---

## 13. Future Enhancements

### Phase 2 (Months 3-6)
- [ ] **Mobile app** (React Native)
- [ ] **Real-time alerts** (SMS/email when χ > 0.8)
- [ ] **Portfolio integration** (connect Alpaca/Interactive Brokers accounts)
- [ ] **Automated trading** (execute trades based on phase-lock signals)
- [ ] **Multi-timeframe analysis** (1min, 5min, 1H, 1D phase-locks)

### Phase 3 (Months 6-12)
- [ ] **Options trading** (phase-lock volatility arbitrage)
- [ ] **Crypto support** (BTC:ETH phase-locks on Binance)
- [ ] **Sentiment analysis** (integrate Twitter/Reddit for χ_social)
- [ ] **Research paper** (publish φ-vortex findings in Journal of Finance)
- [ ] **Hedge fund pilot** (license framework to institutional clients)

---

## 14. Success Metrics

### Technical KPIs
- **Uptime**: 99.9% (less than 43 minutes downtime/month)
- **Response time**: p95 < 2s for chat, p99 < 5s
- **Accuracy**: Phase-lock detection precision > 85%
- **Data latency**: Real-time quotes within 100ms of source

### Business KPIs
- **Month 1**: 100 beta users
- **Month 3**: 500 users, $5K MRR
- **Month 6**: 2,000 users, $50K MRR
- **Month 12**: 10,000 users, $300K MRR

### User Engagement
- **Daily active users**: 40% of total
- **Avg session duration**: > 10 minutes
- **Messages per session**: > 15
- **Backtest runs**: > 2 per user per week
- **Net Promoter Score**: > 50

---

## 15. Conclusion

This architecture provides a **production-ready foundation** for an AI-powered trading research assistant that:

✅ **Integrates cutting-edge research** (φ-vortex framework) with practical application
✅ **Uses best-in-class technologies** (FastAPI, React, TimescaleDB, OpenRouter)
✅ **Scales from MVP to 10K users** with clear upgrade path
✅ **Generates revenue** from day 1 ($49/month subscriptions)
✅ **Provides genuine value** beyond generic ChatGPT clones

**Key Differentiators**:
1. **Novel detection algorithms** (phase-locks, χ-criticality, Fibonacci triads) not available elsewhere
2. **Multi-modal analysis** combining technical indicators with fundamental phase-locking dynamics
3. **Real-time insights** via conversational AI (OpenRouter)
4. **Backtesting** with φ-vortex strategies (Fibonacci pairs, chi reversion, triad arbitrage)
5. **Beautiful visualizations** (D3.js network graphs, heatmaps)

**Next Steps**:
1. **Week 1-2**: Implement MVP (see sprint plan Section 9)
2. **Week 3**: Beta test with 20 users
3. **Week 4**: Launch on Product Hunt, r/algotrading
4. **Month 2**: Iterate based on feedback, add alerts
5. **Month 3**: Scale to 500 users, upgrade to AWS

**The framework is ready. The market is waiting. Time to build.** 🚀

---

**Document Version**: 1.0
**Last Updated**: 2025-11-12
**Author**: φ-Vortex Research Team
**Status**: Ready for Implementation

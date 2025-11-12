# φ-Vortex Trading Assistant: Complete Project Structure

This document provides the complete file structure and quick start guide for implementing the trading assistant.

---

## Project Structure

```
phi-vortex-trading/
│
├── backend/                           # FastAPI backend
│   ├── main.py                        # FastAPI app entry point
│   ├── config.py                      # Environment configuration
│   ├── requirements.txt               # Python dependencies
│   │
│   ├── api/                           # API endpoints
│   │   ├── __init__.py
│   │   ├── chat.py                    # OpenRouter chat endpoint
│   │   ├── phase_locks.py             # Phase-lock detection endpoints
│   │   ├── backtests.py               # Backtesting endpoints
│   │   └── market_data.py             # Market data endpoints
│   │
│   ├── core/                          # φ-Vortex core algorithms
│   │   ├── __init__.py
│   │   ├── phase_lock_detector.py     # Phase-lock detection
│   │   ├── chi_calculator.py          # χ-criticality calculation
│   │   ├── triad_finder.py            # Fibonacci triad detection
│   │   └── constants.py               # PHI, FIBONACCI, etc.
│   │
│   ├── services/                      # Business logic services
│   │   ├── __init__.py
│   │   ├── openrouter_service.py      # OpenRouter API integration
│   │   ├── market_data_service.py     # Market data fetching
│   │   ├── backtest_engine.py         # Backtesting engine
│   │   └── websocket_service.py       # Real-time WebSocket handler
│   │
│   ├── models/                        # SQLAlchemy models
│   │   ├── __init__.py
│   │   ├── market_data.py
│   │   ├── phase_locks.py
│   │   ├── triads.py
│   │   ├── chat_sessions.py
│   │   └── backtests.py
│   │
│   ├── schemas/                       # Pydantic schemas (validation)
│   │   ├── __init__.py
│   │   ├── chat.py
│   │   ├── phase_locks.py
│   │   └── backtests.py
│   │
│   ├── workers/                       # Background tasks (Celery)
│   │   ├── __init__.py
│   │   ├── celery_app.py
│   │   ├── market_data_updater.py     # Daily data fetch
│   │   └── phase_lock_scanner.py      # Periodic lock detection
│   │
│   ├── db/                            # Database
│   │   ├── __init__.py
│   │   ├── session.py                 # SQLAlchemy session
│   │   └── migrations/                # Alembic migrations
│   │       └── versions/
│   │
│   ├── tests/                         # Unit tests (pytest)
│   │   ├── test_phase_lock_detector.py
│   │   ├── test_chi_calculator.py
│   │   ├── test_triad_finder.py
│   │   └── test_api_endpoints.py
│   │
│   └── utils/                         # Utility functions
│       ├── __init__.py
│       ├── auth.py                    # JWT authentication
│       ├── rate_limiting.py
│       └── logging.py
│
├── frontend/                          # React frontend
│   ├── public/
│   │   └── index.html
│   │
│   ├── src/
│   │   ├── App.tsx                    # Main app component
│   │   ├── index.tsx                  # Entry point
│   │   ├── index.css                  # Global styles
│   │   │
│   │   ├── components/                # React components
│   │   │   ├── ChatInterface.tsx      # Main chat UI
│   │   │   ├── PhaseLockGraph.tsx     # D3.js network graph
│   │   │   ├── ChiHeatmap.tsx         # χ-criticality heatmap
│   │   │   ├── CandlestickChart.tsx   # Chart.js candlesticks
│   │   │   ├── TriadVisualization.tsx # Triad display
│   │   │   └── BacktestResults.tsx    # Backtest metrics
│   │   │
│   │   ├── hooks/                     # Custom React hooks
│   │   │   ├── useWebSocket.ts        # WebSocket connection
│   │   │   ├── useChat.ts             # Chat state management
│   │   │   └── useMarketData.ts       # Market data fetching
│   │   │
│   │   ├── services/                  # API clients
│   │   │   ├── api.ts                 # Axios instance
│   │   │   ├── chatService.ts
│   │   │   ├── marketDataService.ts
│   │   │   └── backtestService.ts
│   │   │
│   │   ├── store/                     # State management (Zustand)
│   │   │   ├── chatStore.ts
│   │   │   ├── marketStore.ts
│   │   │   └── userStore.ts
│   │   │
│   │   ├── types/                     # TypeScript types
│   │   │   ├── chat.ts
│   │   │   ├── phaseLocks.ts
│   │   │   └── backtests.ts
│   │   │
│   │   └── utils/                     # Helper functions
│   │       ├── formatters.ts
│   │       └── constants.ts
│   │
│   ├── package.json
│   ├── tsconfig.json
│   ├── tailwind.config.js
│   └── vite.config.ts
│
├── infrastructure/                    # Infrastructure as Code
│   ├── terraform/                     # AWS infrastructure
│   │   ├── main.tf
│   │   ├── vpc.tf
│   │   ├── ecs.tf
│   │   ├── rds.tf
│   │   ├── redis.tf
│   │   └── variables.tf
│   │
│   └── docker/
│       ├── Dockerfile.backend
│       ├── Dockerfile.frontend
│       └── docker-compose.yml         # Local development
│
├── docs/                              # Documentation
│   ├── API.md                         # API documentation
│   ├── PHI_VORTEX_THEORY.md           # Framework explanation
│   └── DEPLOYMENT.md                  # Deployment guide
│
├── scripts/                           # Utility scripts
│   ├── seed_database.py               # Populate with historical data
│   ├── run_backtest.py                # CLI backtest runner
│   └── calculate_metrics.py           # Performance metrics
│
├── .env.example                       # Environment variables template
├── .gitignore
├── README.md
└── LICENSE
```

---

## Quick Start Guide

### Prerequisites

```bash
# Required software
- Python 3.11+
- Node.js 18+
- Docker & Docker Compose
- PostgreSQL 15+ with TimescaleDB
- Redis 7+
```

---

### 1. Clone Repository

```bash
git clone https://github.com/yourusername/phi-vortex-trading.git
cd phi-vortex-trading
```

---

### 2. Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create .env file
cp ../.env.example .env
# Edit .env with your API keys:
# OPENROUTER_API_KEY=sk-or-...
# DATABASE_URL=postgresql://user:pass@localhost:5432/trading
# REDIS_URL=redis://localhost:6379
# POLYGON_API_KEY=...
```

**requirements.txt**:
```txt
fastapi==0.110.0
uvicorn[standard]==0.27.0
sqlalchemy==2.0.0
alembic==1.13.0
psycopg2-binary==2.9.9
redis==5.0.0
celery==5.3.0
httpx==0.26.0
pydantic==2.6.0
python-dotenv==1.0.0
python-jose[cryptography]==3.3.0
passlib[bcrypt]==1.7.4
slowapi==0.1.9
prometheus-client==0.20.0

# Data & ML
numpy==1.26.0
scipy==1.12.0
pandas==2.2.0

# Market data
yfinance==0.2.36
alpaca-trade-api==3.1.1
polygon-api-client==1.13.0

# Testing
pytest==8.0.0
pytest-asyncio==0.23.0
pytest-cov==4.1.0
httpx-ws==0.6.0
```

---

### 3. Database Setup

**Option A: Docker Compose (Recommended for local dev)**:

```bash
# From project root
docker-compose up -d postgres redis

# Wait for PostgreSQL to start (10 seconds)
sleep 10

# Run migrations
cd backend
alembic upgrade head

# Seed with historical data
python scripts/seed_database.py --symbols AAPL,MSFT,GOOGL,META --days 365
```

**docker-compose.yml**:
```yaml
version: '3.8'

services:
  postgres:
    image: timescale/timescaledb:latest-pg15
    environment:
      POSTGRES_USER: trading
      POSTGRES_PASSWORD: changeme
      POSTGRES_DB: trading
    ports:
      - "5432:5432"
    volumes:
      - pgdata:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redisdata:/data

volumes:
  pgdata:
  redisdata:
```

**Option B: Managed Services** (Production):
- Use AWS RDS (PostgreSQL with TimescaleDB)
- Use AWS ElastiCache (Redis)

---

### 4. Run Backend

**Terminal 1 - API Server**:
```bash
cd backend
source venv/bin/activate
uvicorn main:app --reload --port 8000
```

**Terminal 2 - Celery Worker** (background tasks):
```bash
cd backend
source venv/bin/activate
celery -A workers.celery_app worker --loglevel=info
```

**Terminal 3 - Market Data Streamer** (optional, for real-time):
```bash
cd backend
source venv/bin/activate
python workers/market_data_streamer.py
```

Backend will be available at: `http://localhost:8000`
API docs: `http://localhost:8000/docs`

---

### 5. Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Create .env.local
echo "VITE_API_URL=http://localhost:8000" > .env.local

# Run dev server
npm run dev
```

Frontend will be available at: `http://localhost:3000`

**package.json**:
```json
{
  "name": "phi-vortex-trading-frontend",
  "version": "1.0.0",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "tsc && vite build",
    "preview": "vite preview",
    "lint": "eslint src --ext ts,tsx",
    "test": "vitest"
  },
  "dependencies": {
    "react": "^18.3.0",
    "react-dom": "^18.3.0",
    "react-router-dom": "^6.22.0",
    "react-query": "^5.17.0",
    "zustand": "^4.5.0",
    "axios": "^1.6.0",
    "socket.io-client": "^4.7.0",
    "d3": "^7.9.0",
    "chart.js": "^4.4.0",
    "react-chartjs-2": "^5.2.0",
    "lucide-react": "^0.344.0",
    "date-fns": "^3.3.0",
    "clsx": "^2.1.0",
    "tailwind-merge": "^2.2.0"
  },
  "devDependencies": {
    "@types/react": "^18.3.0",
    "@types/react-dom": "^18.3.0",
    "@types/d3": "^7.4.0",
    "@typescript-eslint/eslint-plugin": "^7.0.0",
    "@typescript-eslint/parser": "^7.0.0",
    "@vitejs/plugin-react": "^4.2.0",
    "autoprefixer": "^10.4.0",
    "eslint": "^8.57.0",
    "eslint-plugin-react-hooks": "^4.6.0",
    "postcss": "^8.4.0",
    "tailwindcss": "^3.4.0",
    "typescript": "^5.3.0",
    "vite": "^5.1.0",
    "vitest": "^1.3.0"
  }
}
```

---

### 6. Test the System

**Test 1: Phase-Lock Detection**
```bash
curl http://localhost:8000/api/phase-locks?symbols=AAPL,MSFT
```

Expected response:
```json
{
  "locks": [
    {
      "symbol_a": "AAPL",
      "symbol_b": "MSFT",
      "ratio_m": 2,
      "ratio_n": 1,
      "coupling_strength": 0.85,
      "is_fibonacci": true,
      "stability_days": 3
    }
  ]
}
```

**Test 2: χ-Criticality**
```bash
curl http://localhost:8000/api/chi?symbol=SPY&window_days=30
```

Expected response:
```json
{
  "symbol": "SPY",
  "chi": 0.65,
  "flux": 0.18,
  "dissipation": 0.277,
  "status": "elevated",
  "optimal_chi": 0.382
}
```

**Test 3: Chat with OpenRouter**
```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What phase-locks exist right now?",
    "session_id": "test-session-123"
  }'
```

Expected response (streaming):
```
data: I found 3 strong phase-locks...
data: 1. AAPL:MSFT - 2:1 resonance (Fibonacci!)...
data: ...
```

---

### 7. Seed Database with Historical Data

```bash
cd backend
python scripts/seed_database.py --symbols AAPL,MSFT,GOOGL,META,NVDA,AMD,INTC,TSLA --days 730
```

This will:
1. Fetch 2 years of daily OHLCV data from Yahoo Finance
2. Store in TimescaleDB hypertable
3. Detect all historical phase-locks
4. Compute χ time series
5. Find Fibonacci triads

Expected output:
```
Fetching AAPL... ✓ (730 days)
Fetching MSFT... ✓ (730 days)
...
Detecting phase-locks... ✓ (42 locks found, 31 Fibonacci)
Computing χ time series... ✓ (8 symbols)
Finding triads... ✓ (7 triads found)

Database seeded successfully!
Total records: 5,840 OHLCV, 42 locks, 7 triads
```

---

### 8. Run Tests

**Backend tests**:
```bash
cd backend
pytest tests/ -v --cov=. --cov-report=html

# Expected output:
# test_phase_lock_detector.py::test_detect_2_1_lock PASSED
# test_phase_lock_detector.py::test_fibonacci_detection PASSED
# test_chi_calculator.py::test_chi_calculation PASSED
# test_triad_finder.py::test_find_triads PASSED
# test_api_endpoints.py::test_chat_endpoint PASSED
# ...
# ===================== 47 passed in 3.21s ======================
```

**Frontend tests**:
```bash
cd frontend
npm test

# Expected output:
# ✓ ChatInterface renders correctly
# ✓ PhaseLockGraph displays nodes and edges
# ✓ ChiHeatmap shows correct colors
# ...
# Test Files  12 passed (12)
# Tests  47 passed (47)
```

---

### 9. Deploy to Production

**Railway** (Quick deploy):
```bash
# Backend
cd backend
railway login
railway init
railway add postgresql  # Add Postgres plugin
railway add redis       # Add Redis plugin
railway up

# Frontend
cd frontend
npm run build
vercel --prod
```

**AWS** (Full production):
```bash
cd infrastructure/terraform
terraform init
terraform plan
terraform apply -auto-approve

# Build & push Docker images
cd ../../backend
docker build -t phi-vortex-api:latest -f ../infrastructure/docker/Dockerfile.backend .
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account>.dkr.ecr.us-east-1.amazonaws.com
docker tag phi-vortex-api:latest <account>.dkr.ecr.us-east-1.amazonaws.com/phi-vortex-api:latest
docker push <account>.dkr.ecr.us-east-1.amazonaws.com/phi-vortex-api:latest

# Update ECS service
aws ecs update-service --cluster phi-vortex-cluster --service api --force-new-deployment

# Deploy frontend to CloudFront + S3
cd ../frontend
npm run build
aws s3 sync dist/ s3://phi-vortex-frontend-bucket/
aws cloudfront create-invalidation --distribution-id EXXXXXXXXXXXXX --paths "/*"
```

---

## Environment Variables

**Backend (.env)**:
```bash
# API Keys
OPENROUTER_API_KEY=sk-or-v1-xxxxx
POLYGON_API_KEY=xxxxx
ALPACA_API_KEY=xxxxx
ALPACA_API_SECRET=xxxxx

# Database
DATABASE_URL=postgresql://user:pass@localhost:5432/trading

# Redis
REDIS_URL=redis://localhost:6379

# JWT
JWT_SECRET=your-secret-key-change-in-production
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=1440

# Celery
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/1

# Application
DEBUG=true
LOG_LEVEL=INFO
CORS_ORIGINS=http://localhost:3000,https://yourdomain.com

# Rate Limiting
RATE_LIMIT_PER_MINUTE=60
```

**Frontend (.env.local)**:
```bash
VITE_API_URL=http://localhost:8000
VITE_WS_URL=ws://localhost:8000/ws
VITE_ENV=development
```

---

## Common Issues & Solutions

### Issue 1: Database connection failed
**Error**: `sqlalchemy.exc.OperationalError: could not connect to server`

**Solution**:
```bash
# Check PostgreSQL is running
docker ps | grep postgres

# If not running:
docker-compose up -d postgres

# Check connection
psql -h localhost -U trading -d trading
```

---

### Issue 2: TimescaleDB extension not enabled
**Error**: `ERROR: extension "timescaledb" is not available`

**Solution**:
```sql
-- Connect to database
psql -h localhost -U trading -d trading

-- Enable extension
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Verify
\dx
```

---

### Issue 3: OpenRouter API rate limit
**Error**: `429 Too Many Requests`

**Solution**:
```python
# In backend/services/openrouter_service.py
# Add retry logic with exponential backoff

import asyncio
from tenacity import retry, wait_exponential, stop_after_attempt

@retry(wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(3))
async def call_openrouter(messages, tools):
    # ... API call
```

---

### Issue 4: Phase-lock detection too slow
**Error**: API response > 5 seconds

**Solution**:
```python
# Optimize with caching in Redis
from functools import lru_cache

@lru_cache(maxsize=1000)
def detect_phase_lock_cached(symbol_a, symbol_b, date):
    # Check Redis first
    cached = redis.get(f"lock:{symbol_a}:{symbol_b}:{date}")
    if cached:
        return json.loads(cached)

    # Compute
    result = detect_phase_lock(prices_a, prices_b)

    # Cache for 1 hour
    redis.setex(f"lock:{symbol_a}:{symbol_b}:{date}", 3600, json.dumps(result))
    return result
```

---

### Issue 5: Frontend build fails
**Error**: `Module not found: Can't resolve 'd3'`

**Solution**:
```bash
# Clear node_modules and reinstall
rm -rf node_modules package-lock.json
npm install

# If still fails, check TypeScript version
npm list typescript
# Should be 5.3.0+
```

---

## Performance Optimization

### Backend Optimizations

**1. Database Indexing**:
```sql
-- Add indexes for common queries
CREATE INDEX idx_phase_locks_symbols ON phase_locks(symbol_a, symbol_b);
CREATE INDEX idx_phase_locks_detected_at ON phase_locks(detected_at DESC);
CREATE INDEX idx_market_data_symbol_time ON market_data(symbol, time DESC);

-- Analyze query performance
EXPLAIN ANALYZE SELECT * FROM phase_locks WHERE symbol_a = 'AAPL' AND detected_at > NOW() - INTERVAL '7 days';
```

**2. Connection Pooling**:
```python
# backend/db/session.py
from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool

engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=20,       # Max connections
    max_overflow=10,    # Burst connections
    pool_pre_ping=True  # Check connection health
)
```

**3. Async Endpoints**:
```python
# Use async/await for I/O-bound operations
@app.get("/api/phase-locks")
async def get_phase_locks(symbols: list[str]):
    # Fetch data in parallel
    tasks = [fetch_prices(symbol) for symbol in symbols]
    prices = await asyncio.gather(*tasks)

    # Detect locks
    locks = detect_phase_locks(prices)
    return locks
```

---

### Frontend Optimizations

**1. Code Splitting**:
```typescript
// Lazy load heavy components
import { lazy, Suspense } from 'react';

const PhaseLockGraph = lazy(() => import('./components/PhaseLockGraph'));
const ChiHeatmap = lazy(() => import('./components/ChiHeatmap'));

function App() {
  return (
    <Suspense fallback={<LoadingSpinner />}>
      <PhaseLockGraph />
    </Suspense>
  );
}
```

**2. React Query Caching**:
```typescript
// Cache API responses
import { useQuery } from 'react-query';

function usePhaseLocks(symbols: string[]) {
  return useQuery(
    ['phase-locks', symbols],
    () => fetch(`/api/phase-locks?symbols=${symbols.join(',')}`),
    {
      staleTime: 60000,  // 1 minute
      cacheTime: 300000  // 5 minutes
    }
  );
}
```

**3. WebSocket Batching**:
```typescript
// Batch WebSocket updates to reduce renders
let updateBuffer: MarketUpdate[] = [];
let timer: NodeJS.Timeout;

socket.on('market-update', (update) => {
  updateBuffer.push(update);

  clearTimeout(timer);
  timer = setTimeout(() => {
    setMarketData(prev => [...prev, ...updateBuffer]);
    updateBuffer = [];
  }, 100);  // Batch every 100ms
});
```

---

## Monitoring & Debugging

**View logs**:
```bash
# Backend logs
docker-compose logs -f backend

# Celery logs
docker-compose logs -f celery

# Database logs
docker-compose logs -f postgres
```

**Check metrics**:
```bash
# Prometheus metrics endpoint
curl http://localhost:8000/metrics

# Sample output:
# http_requests_total{method="GET",endpoint="/api/phase-locks",status="200"} 1234
# phase_locks_detected{is_fibonacci="true"} 172
# chi_calculations 5678
```

**Debug phase-lock detection**:
```python
# backend/core/phase_lock_detector.py
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def detect_phase_lock(...):
    logger.debug(f"Testing ratio {m}:{n}")
    logger.debug(f"Order parameter: {order_param}")
    logger.debug(f"Coupling strength: {K_measured}")
    # ...
```

---

## Next Steps

1. ✅ **Set up local environment** (this guide)
2. [ ] **Read architecture doc** (TRADING_ASSISTANT_ARCHITECTURE.md)
3. [ ] **Implement MVP** (2-week sprint)
4. [ ] **Beta test** with 20 users
5. [ ] **Launch** on Product Hunt
6. [ ] **Scale** to 500 users
7. [ ] **Iterate** based on feedback

---

**For questions or issues, open a GitHub issue or contact the team.**

**Happy building! 🚀**

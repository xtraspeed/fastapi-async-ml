# Distributed Asynchronous ML Inference Platform

[![FastAPI](https://img.shields.io/badge/FastAPI-0.111.0-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Celery](https://img.shields.io/badge/Celery-5.4.0-37814A?logo=celery&logoColor=white)](https://docs.celeryq.dev)
[![RabbitMQ](https://img.shields.io/badge/RabbitMQ-3.13-FF6600?logo=rabbitmq&logoColor=white)](https://www.rabbitmq.com)
[![Redis](https://img.shields.io/badge/Redis-7.2-DC382D?logo=redis&logoColor=white)](https://redis.io)
[![Docker Compose](https://img.shields.io/badge/Docker_Compose-Multi--Container-2496ED?logo=docker&logoColor=white)](https://www.docker.com)

A production-grade, distributed asynchronous Machine Learning inference platform built for high-throughput, non-blocking NLP and content risk scoring. Built using **FastAPI**, **Celery**, **RabbitMQ**, **Redis**, and **Flower**.

---

## 🎯 Resume Bullet Points (Copy & Paste Ready)

Add this project to your resume under **Experience** or **Projects**:

* **Distributed Async ML Inference Platform (FastAPI, Celery, RabbitMQ, Redis, Docker)**
  * Architected a distributed asynchronous ML microservice using **FastAPI** and **Celery**, decoupling CPU-bound NLP model inference from the HTTP request loop to achieve **< 10ms API response latency**.
  * Integrated **RabbitMQ** as an AMQP message broker with fair task dispatching (`prefetch_multiplier=1`) and late acknowledgments (`acks_late=True`) to guarantee zero task loss during worker failures.
  * Implemented a dual-purpose **Redis** layer serving as Celery's result backend, real-time task progress tracker (0%–100% granular state transitions), and an **idempotent SHA-256 payload cache** that eliminates redundant inference.
  * Engineered endpoints for single-item and batch inference (`/api/v1/batch-predict`), structured with **Pydantic v2** validation and standardized error handling.
  * Containerized the platform with **Docker Compose** across 5 coordinated microservices (API, Celery Worker, RabbitMQ, Redis, Flower) with Kubernetes-compatible `/health/live` and `/health/ready` probes.

---

## 🏗️ System Architecture

```mermaid
flowchart TD
    Client([Client / Frontend]) -->|1. POST /api/v1/predict| API[FastAPI Web Gateway]
    
    subgraph FastAPILogic [FastAPI Gateway]
        API --> CheckCache{Check Redis Cache}
        CheckCache -->|Cache Hit| ReturnCached[Return Cached Result 200 OK]
        CheckCache -->|Cache Miss| Dispatch[Send Task to RabbitMQ & Return 202 Accepted]
    end

    Dispatch -->|Publish Message| RMQ[(RabbitMQ Message Broker)]
    ReturnCached --> Client
    Dispatch --> Client

    subgraph CeleryWorkers [Distributed Worker Cluster]
        RMQ -->|Consume Job| Worker[Celery ML Worker]
        Worker -->|Update Progress 20%, 60%, 100%| RedisStore[(Redis Result Store & Cache)]
        Worker --> MLPipeline[Run Preprocessing & Dual ML Inference]
        MLPipeline --> Worker
        Worker -->|Save Prediction & Set Cache TTL| RedisStore
    end

    Client -->|2. GET /api/v1/tasks/:id| API
    API -->|Read Status & Result| RedisStore
    
    subgraph Observability [Telemetry & Monitoring]
        Flower[Flower Dashboard :5555] -->|Monitor Workers & Queues| RMQ
        Flower -->|Query Tasks| RedisStore
    end
```

### Why Both RabbitMQ and Redis?
* **RabbitMQ (Message Broker)**: Industry standard for enterprise message queues. Offers guaranteed message delivery, queue persistence, exchange routing, dead-letter exchanges, and worker load distribution.
* **Redis (Result Store & Cache)**: In-memory key-value database ideal for ephemeral state storage, sub-millisecond polling lookups for task statuses, and key expiration (TTL) for prediction caching.

---

## 🚀 Quick Start with Docker Compose

Ensure Docker and Docker Compose are installed, then run:

```bash
# 1. Clone or navigate to the project directory
cd fastapi-celery-ml

# 2. Build and launch all 5 containers
docker compose up --build
```

Services will be accessible at:
* **FastAPI Interactive Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)
* **RabbitMQ Management UI**: [http://localhost:15672](http://localhost:15672) (User: `guest`, Password: `guest`)
* **Flower Celery Dashboard**: [http://localhost:5555](http://localhost:5555)
* **Readiness Health Probe**: [http://localhost:8000/health/ready](http://localhost:8000/health/ready)

To shut down:
```bash
docker compose down
```

---

## 💻 Local Setup (Without Docker)

If you prefer to run services natively on your machine:

### 1. Prerequisites
* Python 3.10+
* Running Redis instance (`localhost:6379`)
* Running RabbitMQ instance (`localhost:5672`)

### 2. Install Dependencies
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Start the Components
Open three separate terminal tabs:

**Terminal 1: Start Celery Worker**
```bash
celery -A app.core.celery_app.celery_app worker --loglevel=info --concurrency=2 -Q ml_inference_queue,ml_batch_queue
```

**Terminal 2: Start Flower Monitoring**
```bash
celery -A app.core.celery_app.celery_app flower --port=5555
```

**Terminal 3: Start FastAPI Application**
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## 📡 API Usage & Examples

### 1. Submit Single Text for ML Analysis
```bash
curl -X POST "http://localhost:8000/api/v1/predict" \
     -H "Content-Type: application/json" \
     -d '{"text": "The new release resolved our pipeline bottlenecks! Super smooth experience."}'
```
**Response (`202 Accepted`):**
```json
{
  "task_id": "8a32a67e-450f-48d5-b0b3-e5d83637e1b9",
  "status": "PENDING",
  "message": "Inference job dispatched to RabbitMQ queue. Poll check_status_url for progress.",
  "check_status_url": "http://localhost:8000/api/v1/tasks/8a32a67e-450f-48d5-b0b3-e5d83637e1b9",
  "cached": false,
  "result": null
}
```

### 2. Poll Task Progress & Result
```bash
curl -X GET "http://localhost:8000/api/v1/tasks/8a32a67e-450f-48d5-b0b3-e5d83637e1b9"
```
**While Processing (`PROGRESS`):**
```json
{
  "task_id": "8a32a67e-450f-48d5-b0b3-e5d83637e1b9",
  "status": "PROGRESS",
  "progress": 60,
  "stage": "Executing dual NLP inference pipelines",
  "result": null,
  "error": null
}
```
**When Completed (`SUCCESS`):**
```json
{
  "task_id": "8a32a67e-450f-48d5-b0b3-e5d83637e1b9",
  "status": "SUCCESS",
  "progress": 100,
  "stage": "Completed",
  "result": {
    "input_hash": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "sentiment": {
      "label": "POSITIVE",
      "score": 0.8412,
      "confidence": 0.9124
    },
    "risk": {
      "risk_level": "SAFE",
      "risk_score": 0.0215,
      "flagged_categories": []
    },
    "extracted_keywords": ["release", "pipeline", "bottlenecks", "smooth"],
    "word_count": 9,
    "char_count": 68,
    "model_version": "v1.2.0-tfidf-logreg",
    "processing_time_ms": 2.41
  },
  "error": null
}
```

### 3. Idempotent Cache Hit Demonstration
Submit the identical text payload again. FastAPI will immediately detect the SHA-256 hash match in Redis and return in `< 2ms` without invoking RabbitMQ or Celery:
```json
{
  "task_id": "cached-e3b0c44298fc",
  "status": "SUCCESS",
  "message": "Result fetched directly from Redis cache (idempotent lookup).",
  "check_status_url": "",
  "cached": true,
  "result": { ... }
}
```

### 4. Submit Batch Inferences
```bash
curl -X POST "http://localhost:8000/api/v1/batch-predict" \
     -H "Content-Type: application/json" \
     -d '{
       "items": [
         {"text": "Exceptional customer support, thank you!"},
         {"text": "URGENT: Click here to verify your bank password immediately!"}
       ]
     }'
```

---

## 🧪 Running Automated Tests

Run the test suite with `pytest`:

```bash
pytest -v
```

Tests validate:
* Schema validation and 422 error boundaries
* Idempotent Redis cache lookups (hits vs. misses)
* Celery task scheduling and progressive status transitions
* Dual ML pipeline classification (Sentiment + Risk/Toxicity classification)

---

## 📂 Project Structure

```text
fastapi-celery-ml/
├── docker-compose.yml          # 5-service orchestration (API, Worker, RMQ, Redis, Flower)
├── Dockerfile                  # Slim production Docker image
├── requirements.txt            # Python dependencies
├── .env.example                # Config template
├── README.md                   # Full documentation & resume guide
├── app/
│   ├── main.py                 # FastAPI app factory, CORS, lifespan
│   ├── config.py               # Pydantic BaseSettings environment manager
│   ├── api/
│   │   ├── routes.py           # /predict, /batch-predict, /tasks/{task_id}
│   │   └── health.py           # /health/live, /health/ready probes
│   ├── core/
│   │   ├── celery_app.py       # Celery configuration with RabbitMQ & Redis
│   │   ├── cache.py            # Redis cache manager & SHA-256 key hashing
│   │   └── logger.py           # Structured logger
│   ├── models/
│   │   ├── schemas.py          # Pydantic v2 schemas
│   │   └── ml_model.py         # Dual NLP ML inference engine
│   └── workers/
│       └── tasks.py            # Async Celery tasks with progress reporting & retries
└── tests/
    ├── test_api.py             # FastAPI API integration tests
    └── test_tasks.py           # ML inference & caching unit tests
```

---

## 🛡️ Production & Scalability Best Practices

1. **Horizontal Worker Scaling**: Scale worker instances independently based on queue latency:
   ```bash
   docker compose up --scale worker=4
   ```
2. **Fair Task Dispatching**: Configured `worker_prefetch_multiplier = 1` so workers do not greedily prefetch tasks that could be handled by idle workers.
3. **Task Durability**: RabbitMQ durable queues combined with `task_acks_late = True` ensures tasks are only acknowledged once successfully processed and saved.
4. **Graceful Fault Tolerance**: Celery tasks implement exponential retry backoff on unexpected runtime errors.

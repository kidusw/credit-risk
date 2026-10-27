# credit-risk

Credit risk and loan default: PD, scorecards, expected loss, reason codes.

## Setup

```powershell
uv sync                         # all dependency groups for local development
copy .env.example .env          # then fill in KAGGLE_API_TOKEN
docker compose up -d db         # PostgreSQL 16 on localhost:5434
uv run crisk migrate
uv run pytest
```

Behind a TLS-intercepting proxy, set `UV_SYSTEM_CERTS=1`, `REQUESTS_CA_BUNDLE` and
`SSL_CERT_FILE` to a CA bundle that includes the proxy's root, and pass the same bundle to
image builds with `CA_BUNDLE=<path> docker compose build`.

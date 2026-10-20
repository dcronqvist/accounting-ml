# accounting-ml

Lightweight transaction categorization using FastAPI, `HashingVectorizer`, and
`SGDClassifier`. Predicts account numbers from descriptions and amounts, and
learns incrementally from corrections.

## Self-host with Docker Compose

Create a `compose.yml` on your server:

```yaml
services:
  app:
    image: ghcr.io/dcronqvist/accounting-ml:latest
    restart: unless-stopped
    ports:
      - "127.0.0.1:8000:8000"
    volumes:
      - ./data:/data
```

Create the `data` directory alongside it and optionally copy your seed CSV to
`data/transactions.csv` over SCP. Then run:

```sh
docker compose pull
docker compose up -d
docker compose logs -f
```

Images support **Linux AMD64**. If the GHCR package is private, first run
`docker login ghcr.io -u YOUR_USERNAME` and enter a personal access token (classic)
with `read:packages` at the password prompt.

The API is at http://localhost:8000; interactive docs are at
http://localhost:8000/docs. There is **no authentication**: keep the loopback
binding and use an SSH tunnel, or add an authenticated HTTPS reverse proxy.

To upgrade, run `docker compose pull && docker compose up -d`. The data directory
persists across container replacements. Pin a version tag or digest instead of
`latest` for controlled upgrades.

## Seed data and persistence

The seed must be a UTF-8 CSV:

```csv
description,amount,account_number
Groceries,54.20,0400
Rent,1200.00,0301
,100.00,0400
```

- Use consistent currency and signs; the current dataset uses positive amounts.
- The seed is imported only when `/data/model.pkl` does not exist. Without a
  seed, submit `/update` first; untrained predictions return HTTP 503.
- Empty, null, omitted, or whitespace-only API descriptions become
  `no_description`. Blank CSV descriptions use the same marker.
- Updates save automatically. New accounts trigger a rebuild from retained
  examples; known accounts use `partial_fit`. With one account, all predictions
  return that account.
- To retrain from a replacement CSV, stop the app, back up and remove
  `data/model.pkl`, then start it again.

**Keep `data` private and backed up.** Snapshots contain training transactions,
not just weights. CSVs and models are excluded from Git and the image. Only load
trusted snapshots, and retrain from CSV when changing incompatible dependency
versions. Run only **one worker/container per model file**.

## API

Predict:

```sh
curl -sS http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"description":"Groceries","amount":54.20}'
```

Response: `{"account_number":"0400"}` (depends on training).

Correct and learn:

```sh
curl -sS http://localhost:8000/update \
  -H 'Content-Type: application/json' \
  -d '{"description":"Groceries","amount":54.20,"correct_account_number":"0400"}'
```

Response: `{"status":"updated","account_number":"0400"}`.
Descriptions are optional; `{"amount":100}` is a valid prediction request.
Account numbers are strings, preserving leading zeros.

## Development and publishing

Build locally with the repository's `docker-compose.yml`:
`docker compose up --build`.

For tests, install `requirements.txt` in a virtual environment and run
`python -m unittest -v`. Tests use temporary storage.

GitHub Actions publishes `latest` for relevant `main` changes, `v*` version tags,
and `sha-<full-commit-sha>` tags. Manual publishing is also available. Builds use
layer caching and skip tests and ARM64 emulation to minimize runner time.
Documentation-only changes do not trigger publishing; run tests locally before
pushing application changes.

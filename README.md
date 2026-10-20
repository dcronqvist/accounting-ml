# Real-time transaction categorization

A small FastAPI service using hashed description features, a signed log-scaled
amount feature, and `SGDClassifier(loss="log_loss")`. Account numbers are strings,
so leading zeros are preserved.

## Start with Docker (Windows / PowerShell)

Install Docker Desktop with Linux containers enabled. Create a `data` folder
next to this README:

```powershell
New-Item -ItemType Directory -Force .\data
```

Optionally place a UTF-8 `transactions.csv` in that folder:

```csv
description,amount,account_number
Grocery supermarket,-54.20,0400
Weekly groceries,-32.10,0400
Monthly salary,3200.00,1000
Employer payroll,3100.00,1000
```

Use consistent signs and currency for amounts; this service does not convert
currencies. Quote descriptions containing commas using standard CSV quoting.

The current seed generated from the transaction export uses negative rows
immediately followed by an equal, opposite positive amount. It takes the
description from the negative row, and the account number and positive amount
from the positive row. Missing descriptions are represented by `no_description`.
Both endpoints automatically normalize omitted, null, empty, or whitespace-only
descriptions to this marker. Blank seed CSV descriptions are normalized too. Use
positive amounts to match this seed. The placeholder does not increase the
amount feature's weight; it is a shared text feature for transactions without
descriptions. Amount alone may not reliably distinguish accounts.

```powershell
docker compose up --build
```

The API is available at http://localhost:8000 and interactive documentation
(also useful instead of Postman) at http://localhost:8000/docs.

`./data:/data` is a bind mount: the seed CSV, model, and retained training examples
remain on the Windows host after container rebuilds or `docker compose down`.
The container needs write access to this folder. The seed is imported only when
no trained snapshot exists. Invalid CSV data or an unreadable/corrupt snapshot
fails startup rather than silently discarding training.

## Test the API

PowerShell examples (use `curl.exe`, not the PowerShell `curl` alias):

```powershell
'{"description":"Grocery supermarket","amount":-42.50}' | curl.exe -sS --json '@-' http://localhost:8000/predict
```

Example response: `{"account_number":"0400"}`. Predictions depend on training
data; this tiny seed is illustrative, not an accuracy guarantee.

```powershell
'{"description":"Grocery supermarket","amount":-42.50,"correct_account_number":"0400"}' | curl.exe -sS --json '@-' http://localhost:8000/update
```

Response: `{"status":"updated","account_number":"0400"}`.

For older curl versions, replace `--json '@-'` with
`-H 'Content-Type: application/json' --data-binary '@-'`.
In Postman, send POST requests to the same URLs with a raw JSON body.

Without a seed, call `/update` to start learning. `/predict` returns HTTP 503
until the first example is recorded. With only one known account it always
returns that account; actual SGD training begins at two distinct accounts.
Updates for existing accounts use `partial_fit`. A new account triggers a rebuild
from all retained examples because scikit-learn requires a fixed class set per
classifier. All updates are atomically saved before success is returned.

For example, `{"amount":100}` and `{"description":"","amount":100}` are both
valid prediction requests.

Blank account numbers, invalid description types, descriptions over 2,000
characters, non-finite amounts, missing required fields, and unexpected fields
return HTTP 422. Persistence failures return HTTP 500 and are
logged; the in-memory model remains unchanged.

## Local development and tests

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
$env:MODEL_PATH = "$PWD\data\model.pkl"
$env:SEED_CSV_PATH = "$PWD\data\transactions.csv"
.\.venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8000
```

In another terminal:

```powershell
.\.venv\Scripts\python.exe -m unittest -v
```

The tests use temporary storage and require no extra test dependencies.

## Publish an image with GitHub Actions

The workflow in `.github/workflows/publish-image.yml` is ready for when you
initialize Git and push this project to a GitHub repository yourself. No
repository initialization or publication is needed to prepare these files.

1. Create your repository and push the project to its default branch.
2. In the repository's **Actions** tab, enable Actions if prompted. Ensure your
   repository/organization policy permits workflows to write packages.
3. The **Publish Docker image** workflow tests the app, then builds and pushes
   Linux AMD64 and ARM64 images to `ghcr.io/OWNER/REPOSITORY` (all lowercase).
   It authenticates using GitHub's automatic `GITHUB_TOKEN`; no additional
   publishing secret is required.

The workflow runs for default-branch pushes and tags beginning with `v`.
Other branch pushes are skipped. You can also run it manually from the Actions
tab with the default branch selected.

Published tags:

- `latest`: the most recently published default-branch build.
- `v1.0.0`, for example: a matching version tag you push.
- `sha-<full-commit-sha>`: a commit-specific image tag.

Version-tag builds do not replace `latest`. For controlled server upgrades, use
a version/commit tag or pin the image digest shown in the workflow build output.

**Keep financial data private:** CSV files, pickle snapshots, and `data/` are
excluded from Git and the Docker build context. The image contains application
code, not your transactions or trained model. Do not force-add private data to
Git. Transfer the seed to your server separately over a secure connection.

### Pull and run on your server

Replace `owner/repository` with your lowercase GitHub owner and repository name.
For a private package, authenticate first using a personal access token
(classic) with `read:packages`, authorized for your organization if required.
Use `docker login ghcr.io -u YOUR_GITHUB_USERNAME --password-stdin` and supply
the token on standard input via your server's secret-management mechanism.
Do not put tokens in this repository or directly in shell command arguments.
For unauthenticated pulls, explicitly make the package public in its GitHub
package settings; do not assume it is public just because the repository is.

For a Windows server running Linux containers, the following PowerShell example
pulls and starts the image. On a Linux server, use the equivalent Docker commands
with an absolute host data directory and your shell's syntax.

```powershell
New-Item -ItemType Directory -Force .\data
docker pull ghcr.io/owner/repository:latest
docker run -d --name transaction-categorizer --restart unless-stopped `
  -p 127.0.0.1:8000:8000 `
  --mount "type=bind,source=$PWD\data,target=/data" `
  ghcr.io/owner/repository:latest
```

Place your seed at `data/transactions.csv` on the server before starting the
container. It will train and persist its own `data/model.pkl`. An existing model
is loaded instead of retraining. Prefer training on the server from the CSV over
copying a snapshot created with different dependency versions.

Port 8000 is bound to loopback deliberately: access it locally or through an SSH
tunnel. Remote access needs an authenticated HTTPS reverse proxy; this API has
no built-in authentication.

To upgrade, pull the desired tag, stop and remove only the old application
container, then recreate it with the same data mount. Your data remains on the
host. Back it up before upgrades and keep one running container per model file.

## Operational limits

- Run **one Uvicorn worker / one container per model file**. A thread lock
  serializes requests inside that process; multiple processes would overwrite
  each other's updates. Do not enable reload in production.
- Training examples (including descriptions) are retained inside `model.pkl`
  to support new accounts. Storage, rebuild time, and snapshot write time grow
  with history; this lightweight scaffold is for small personal datasets.
  Repeated corrections are additional training examples, not edits to old rows.
- Keep backups of `data`, and restrict access: it contains financial data.
  Joblib/pickle snapshots can execute code when loaded; load only files you trust.
  Keep the same dependency versions when restoring snapshots, or retrain from
  the original data after upgrading scikit-learn.
- No authentication is included. Do not expose this API to untrusted networks.
  For local-only Docker access, change the port mapping to
  `"127.0.0.1:8000:8000"`. Add authentication/TLS before remote deployment.
- Predictions are suggestions, not financial advice or automatic bookkeeping.

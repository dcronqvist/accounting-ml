import copy
import logging
import math
import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from threading import RLock

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier

logger = logging.getLogger(__name__)


class Transaction(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    description: str = Field(default="no_description", min_length=1, max_length=2000)
    amount: float

    @field_validator("description", mode="before", json_schema_input_type=str | None)
    @classmethod
    def strip_description(cls, value: object) -> object:
        if value is None:
            return "no_description"
        if isinstance(value, str):
            return value.strip() or "no_description"
        return value


class Correction(Transaction):
    correct_account_number: str = Field(min_length=1, max_length=100)

    @field_validator("correct_account_number")
    @classmethod
    def strip_account_number(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("correct_account_number must not be blank")
        return value.strip()


class Prediction(BaseModel):
    account_number: str


class UpdateResult(BaseModel):
    status: str
    account_number: str


class ModelNotTrainedError(RuntimeError):
    pass


class TransactionClassifier:
    """Single-process online classifier with atomic, durable snapshots."""

    def __init__(self, model_path: Path | str = "/data/model.pkl"):
        self.model_path = Path(model_path)
        self.lock = RLock()
        self.vectorizer = HashingVectorizer(
            n_features=2**18, alternate_sign=False, ngram_range=(1, 2)
        )
        self.classifier = SGDClassifier(loss="log_loss", random_state=42)
        self.examples: list[Correction] = []
        self.accounts: list[str] = []
        if self.model_path.exists():
            # Only load snapshots from a trusted, locally controlled directory.
            state = joblib.load(self.model_path)
            if not isinstance(state, dict) or state.get("version") != 1:
                raise ValueError("Unsupported model snapshot format")
            self.vectorizer = state["vectorizer"]
            self.classifier = state["classifier"]
            self.examples = [
                Correction.model_validate(example) for example in state["examples"]
            ]
            self.accounts = sorted(
                {example.correct_account_number for example in self.examples}
            )
            if not self.accounts:
                raise ValueError("Model snapshot contains no training examples")
            logger.info("Loaded model with %d accounts", len(self.accounts))

    @property
    def is_trained(self) -> bool:
        return bool(self.accounts)

    def _features(self, examples: list[Transaction]):
        text = self.vectorizer.transform([item.description for item in examples])
        # Compress monetary outliers while preserving amount sign and magnitude.
        amounts = csr_matrix(
            [
                [math.copysign(math.log1p(abs(item.amount)), item.amount) / 10]
                for item in examples
            ]
        )
        return hstack([text, amounts], format="csr")

    def predict(self, description: str | None, amount: float) -> str:
        transaction = Transaction(description=description, amount=amount)
        with self.lock:
            if not self.is_trained:
                raise ModelNotTrainedError(
                    "No training data. Supply transactions.csv or call /update first."
                )
            if len(self.accounts) == 1:
                return self.accounts[0]
            return str(self.classifier.predict(self._features([transaction]))[0])

    def _save(self, classifier: SGDClassifier, examples: list[Correction]) -> None:
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=self.model_path.parent, prefix=".model-", suffix=".tmp", delete=False
            ) as temporary:
                temporary_path = Path(temporary.name)
                joblib.dump(
                    {
                        "version": 1,
                        "vectorizer": self.vectorizer,
                        "classifier": classifier,
                        "examples": [item.model_dump() for item in examples],
                    },
                    temporary,
                )
                temporary.flush()
                os.fsync(temporary.fileno())
            os.replace(temporary_path, self.model_path)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    def _train(self, additions: list[Correction]) -> None:
        examples = self.examples + additions
        accounts = sorted({item.correct_account_number for item in examples})
        if accounts != self.accounts:
            candidate = SGDClassifier(loss="log_loss", random_state=42)
            if len(accounts) >= 2:
                candidate.partial_fit(
                    self._features(examples),
                    [item.correct_account_number for item in examples],
                    classes=accounts,
                )
        else:
            # Keep the live model unchanged if training or persistence fails.
            candidate = copy.deepcopy(self.classifier)
            if len(accounts) >= 2:
                candidate.partial_fit(
                    self._features(additions),
                    [item.correct_account_number for item in additions],
                )
        self._save(candidate, examples)
        self.classifier = candidate
        self.examples = examples
        self.accounts = accounts

    def update(
        self, description: str | None, amount: float, correct_account_number: str
    ) -> None:
        example = Correction(
            description=description,
            amount=amount,
            correct_account_number=correct_account_number,
        )
        with self.lock:
            self._train([example])

    def bootstrap(self, csv_path: Path | str) -> None:
        with self.lock:
            if self.is_trained or not Path(csv_path).exists():
                return
            frame = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
            required = {"description", "amount", "account_number"}
            if not required.issubset(frame.columns):
                raise ValueError(
                    "Seed CSV must contain description, amount, account_number columns"
                )
            examples = []
            for row_number, row in enumerate(frame.to_dict(orient="records"), start=2):
                try:
                    examples.append(
                        Correction(
                            description=row["description"],
                            amount=row["amount"],
                            correct_account_number=row["account_number"],
                        )
                    )
                except ValueError as exc:
                    raise ValueError(f"Invalid seed CSV row {row_number}: {exc}") from exc
            if not examples:
                raise ValueError("Seed CSV contains no transactions")
            self._train(examples)
            logger.info("Bootstrapped model from %d seed transactions", len(examples))


@asynccontextmanager
async def lifespan(app: FastAPI):
    classifier = TransactionClassifier(os.getenv("MODEL_PATH", "/data/model.pkl"))
    classifier.bootstrap(os.getenv("SEED_CSV_PATH", "/data/transactions.csv"))
    app.state.classifier = classifier
    yield


app = FastAPI(title="Transaction categorization", lifespan=lifespan)


@app.post("/predict", response_model=Prediction)
def predict(transaction: Transaction) -> Prediction:
    try:
        account = app.state.classifier.predict(
            transaction.description, transaction.amount
        )
    except ModelNotTrainedError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return Prediction(account_number=account)


@app.post("/update", response_model=UpdateResult)
def update(correction: Correction) -> UpdateResult:
    try:
        app.state.classifier.update(
            correction.description, correction.amount, correction.correct_account_number
        )
    except OSError as exc:
        logger.exception("Failed to persist transaction model")
        raise HTTPException(
            status_code=500, detail="Model could not be saved; update was not applied."
        ) from exc
    return UpdateResult(
        status="updated", account_number=correction.correct_account_number
    )

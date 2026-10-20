import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from main import ModelNotTrainedError, Transaction, TransactionClassifier


class ClassifierTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "model.pkl"
        self.model = TransactionClassifier(self.path)

    def test_cold_start_single_class_and_reload(self):
        with self.assertRaises(ModelNotTrainedError):
            self.model.predict("Groceries", -20)
        self.model.update("Groceries", -20, "0400")
        self.assertEqual(self.model.predict("Food", -10), "0400")
        reloaded = TransactionClassifier(self.path)
        self.assertEqual(reloaded.predict("Food", -10), "0400")

    def test_incremental_learning_and_new_account_rebuild(self):
        self.model.update("Groceries", -20, "0400")
        self.model.update("Salary", 3000, "1000")
        self.assertEqual(self.model.classifier.loss, "log_loss")
        iterations = self.model.classifier.t_
        self.model.update("Weekly groceries", -30, "0400")
        self.assertEqual(self.model.classifier.t_, iterations + 1)
        self.model.update("Train ticket", -15, "0600")
        self.assertEqual(list(self.model.classifier.classes_), ["0400", "0600", "1000"])
        reloaded = TransactionClassifier(self.path)
        self.assertEqual(len(reloaded.examples), 4)
        self.assertEqual(
            reloaded.predict("Train ticket", -15),
            self.model.predict("Train ticket", -15),
        )

    def test_features_include_amount(self):
        positive = self.model._features(
            [Transaction(description="Transfer", amount=100)]
        )
        negative = self.model._features(
            [Transaction(description="Transfer", amount=-100)]
        )
        self.assertGreater(positive[0, -1], 0)
        self.assertEqual(positive[0, -1], -negative[0, -1])
        self.assertEqual((positive[:, :-1] != negative[:, :-1]).nnz, 0)

    def test_missing_descriptions_normalize_and_persist(self):
        self.assertEqual(Transaction(amount=100).description, "no_description")
        for description in (None, "", " \t\n", "no_description"):
            with self.subTest(description=description):
                self.model.update(description, 100, "0400")
                self.assertEqual(self.model.examples[-1].description, "no_description")
                self.assertEqual(self.model.predict(description, 100), "0400")
        self.assertTrue(
            all(
                item.description == "no_description"
                for item in TransactionClassifier(self.path).examples
            )
        )
        self.assertEqual(
            Transaction(description="  Groceries  ", amount=100).description,
            "Groceries",
        )

    def test_csv_blank_description_normalizes(self):
        seed = Path(self.directory.name) / "transactions.csv"
        seed.write_text(
            "description,amount,account_number\n,100,0400\n   ,200,1000\n",
            encoding="utf-8",
        )
        self.model.bootstrap(seed)
        self.assertTrue(
            all(item.description == "no_description" for item in self.model.examples)
        )

    def test_csv_preserves_account_numbers_and_is_only_loaded_once(self):
        seed = Path(self.directory.name) / "transactions.csv"
        seed.write_text(
            "description,amount,account_number\n"
            "Groceries,-20,0400\nSalary,3000,1000\n",
            encoding="utf-8",
        )
        self.model.bootstrap(seed)
        self.assertEqual(self.model.accounts, ["0400", "1000"])
        seed.write_text("invalid", encoding="utf-8")
        self.model.bootstrap(seed)
        self.assertEqual(len(self.model.examples), 2)

    def test_invalid_seed_does_not_create_snapshot(self):
        seed = Path(self.directory.name) / "transactions.csv"
        for contents in (
            "description,amount\nGroceries,-20\n",
            "description,amount,account_number\nGroceries,no,0400\n",
            "description,amount,account_number\nGroceries,inf,0400\n",
            "description,amount,account_number\n",
        ):
            with self.subTest(contents=contents):
                seed.write_text(contents, encoding="utf-8")
                with self.assertRaises(ValueError):
                    self.model.bootstrap(seed)
                self.assertFalse(self.model.is_trained)
                self.assertFalse(self.path.exists())

    def test_failed_save_preserves_live_and_disk_state(self):
        self.model.update("Groceries", -20, "0400")
        self.model.update("Salary", 3000, "1000")
        old_bytes = self.path.read_bytes()
        old_iterations = self.model.classifier.t_
        with patch("main.os.replace", side_effect=OSError("disk unavailable")):
            with self.assertRaises(OSError):
                self.model.update("Groceries", -30, "0400")
        self.assertEqual(self.model.classifier.t_, old_iterations)
        self.assertEqual(len(self.model.examples), 2)
        self.assertEqual(self.path.read_bytes(), old_bytes)
        self.assertEqual(list(self.path.parent.glob(".model-*.tmp")), [])

    def test_corrupt_snapshot_fails_explicitly(self):
        self.path.write_bytes(b"not a snapshot")
        with self.assertRaises(Exception):
            TransactionClassifier(self.path)


class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.directory.cleanup)
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        cls.base_url = f"http://127.0.0.1:{port}"
        env = {
            **os.environ,
            "MODEL_PATH": str(Path(cls.directory.name) / "model.pkl"),
            "SEED_CSV_PATH": str(Path(cls.directory.name) / "absent.csv"),
        }
        cls.logs = tempfile.TemporaryFile(mode="w+b")
        cls.addClassCleanup(cls.logs.close)
        cls.process = subprocess.Popen(
            [
                sys.executable, "-m", "uvicorn", "main:app",
                "--host", "127.0.0.1", "--port", str(port),
            ],
            cwd=Path(__file__).parent,
            env=env,
            stdout=cls.logs,
            stderr=subprocess.STDOUT,
        )
        cls.addClassCleanup(cls.stop_server)
        for _ in range(100):
            try:
                with urlopen(f"{cls.base_url}/openapi.json", timeout=1):
                    return
            except (URLError, TimeoutError):
                if cls.process.poll() is not None:
                    break
                time.sleep(0.1)
        cls.logs.seek(0)
        raise RuntimeError(f"API failed to start: {cls.logs.read().decode()}")

    @classmethod
    def stop_server(cls):
        if cls.process.poll() is None:
            cls.process.terminate()
            try:
                cls.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                cls.process.kill()
                cls.process.wait(timeout=10)

    def post(self, endpoint, body):
        request = Request(
            self.base_url + endpoint,
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            response = urlopen(request, timeout=10)
        except HTTPError as exc:
            response = exc
        with response:
            return response.status, json.load(response)

    def test_http_workflow_and_validation(self):
        transaction = {"description": "Groceries", "amount": -25}
        self.assertEqual(self.post("/predict", transaction)[0], 503)
        status, body = self.post(
            "/update", {**transaction, "correct_account_number": "0400"}
        )
        self.assertEqual((status, body["status"]), (200, "updated"))
        self.assertEqual(
            self.post("/predict", transaction),
            (200, {"account_number": "0400"}),
        )
        self.assertEqual(
            self.post("/update", {
                "description": "Salary", "amount": 3000,
                "correct_account_number": "1000",
            })[0], 200,
        )
        status, body = self.post("/predict", transaction)
        self.assertEqual(status, 200)
        self.assertIn(body["account_number"], ["0400", "1000"])
        for transaction_without_description in (
            {"amount": 100},
            {"description": None, "amount": 100},
            {"description": "", "amount": 100},
            {"description": " \t\n", "amount": 100},
        ):
            with self.subTest(transaction=transaction_without_description):
                self.assertEqual(
                    self.post("/update", {
                        **transaction_without_description,
                        "correct_account_number": "0400",
                    })[0],
                    200,
                )
                status, result = self.post("/predict", transaction_without_description)
                self.assertEqual(status, 200)
                self.assertIn(result["account_number"], ["0400", "1000"])
        restored = TransactionClassifier(Path(self.directory.name) / "model.pkl")
        self.assertEqual(
            [item.description for item in restored.examples[-4:]],
            ["no_description"] * 4,
        )
        for invalid in (
            {"description": 123, "amount": 1},
            {"description": [], "amount": 1},
            {"description": "x" * 2001, "amount": 1},
            {"description": "Groceries", "amount": "NaN"},
            {"description": "Groceries", "amount": "Infinity"},
            {"description": "Groceries"},
            {**transaction, "unexpected": True},
        ):
            with self.subTest(invalid=invalid):
                self.assertEqual(self.post("/predict", invalid)[0], 422)
                self.assertEqual(
                    self.post("/update", {
                        **invalid, "correct_account_number": "0400",
                    })[0],
                    422,
                )
        self.assertEqual(
            self.post("/update", {**transaction, "correct_account_number": " "})[0],
            422,
        )
        self.assertTrue((Path(self.directory.name) / "model.pkl").exists())


if __name__ == "__main__":
    unittest.main()

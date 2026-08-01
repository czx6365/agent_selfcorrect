from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import main
from eval.export_logs import export_logs
from eval.metrics import exact_match, extract_gsm8k_answer
from tools.calculator import calculate


class FakeClient:
    model = "fake-model"
    max_tokens = 128

    def complete(self, prompt: str) -> str:
        return "2 + 3 = 5\nFINAL: 5"


class CoreBehaviorTests(unittest.TestCase):
    def test_extract_gsm8k_answer_handles_final_and_fractions(self) -> None:
        self.assertEqual(extract_gsm8k_answer("work\nFINAL: 1/2"), "0.5")
        self.assertEqual(extract_gsm8k_answer("answer is $1,234"), "1234")
        self.assertTrue(exact_match("12.0", "12"))

    def test_calculator_allows_arithmetic_only(self) -> None:
        self.assertEqual(calculate("(3 + 2) * 4"), "20")
        with self.assertRaises(ValueError):
            calculate("__import__('os').system('echo bad')")

    def test_solve_command_writes_trace_without_real_llm(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            log_path = Path(temp_dir) / "solve_trace.jsonl"
            output = io.StringIO()
            with patch.object(main, "build_client", return_value=FakeClient()):
                with redirect_stdout(output):
                    main.solve_main(
                        [
                            "小明有2个苹果，又买了3个，一共有几个？",
                            "--method",
                            "baseline",
                            "--question-id",
                            "unit_case",
                            "--log",
                            str(log_path),
                        ]
                    )

            self.assertEqual(output.getvalue().strip(), "5")
            record = json.loads(log_path.read_text(encoding="utf-8"))
            self.assertEqual(record["id"], "unit_case")
            self.assertEqual(record["method"], "baseline_cot")
            self.assertEqual(record["prediction"], "5")
            self.assertEqual(record["trace"]["final_answer"], "5")

    def test_export_logs_normalizes_math_and_code_records(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            math_path = root / "math.jsonl"
            code_path = root / "code.jsonl"
            output_path = root / "solve_trace.jsonl"
            math_path.write_text(
                json.dumps(
                    {
                        "id": "gsm8k_test_000",
                        "method": "baseline_cot",
                        "question": "1 + 1?",
                        "prediction": "2",
                        "correct": True,
                        "trace": {"steps": [{"feedback_source": "none"}]},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            code_path.write_text(
                json.dumps(
                    {
                        "id": "HumanEval/0",
                        "method": "code_self_repair_r3",
                        "entry_point": "f",
                        "passed": False,
                        "status": "error",
                        "trace": {"steps": []},
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            count = export_logs(
                math_records=math_path,
                code_records=code_path,
                output=output_path,
                min_method_records=1,
            )

            self.assertEqual(count, 1)
            exported = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(exported["task_type"], "math")
            self.assertEqual(exported["method"], "baseline_cot")


if __name__ == "__main__":
    unittest.main()

# Vendored from NVIDIA/NeMo-Skills@645cf567ff08c0ae9cc3fc8e1edbb975b3067816
# Source: nemo_skills/evaluation/evaluator/math.py
# DO NOT EDIT directly. To upgrade, edit SOURCES.yaml and rerun
# `python scripts/sync_vendored.py`.

# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from dataclasses import field

from sgl_eval._vendored.nemo_skills.evaluator.base import BaseEvaluator, BaseEvaluatorConfig
from sgl_eval._vendored.nemo_skills.math_grader import extract_answer, math_equal
from sgl_eval._vendored.nemo_skills._utils import get_logger_name, nested_dataclass

LOG = logging.getLogger(get_logger_name(__file__))


@nested_dataclass(kw_only=True)
class MathEvaluatorConfig(BaseEvaluatorConfig):
    numeric_precision: int = 15
    timeout: int = 10
    # if True will not attempt to re-extract based on \boxed or regex
    use_predicted_answer_key: bool = False

    extract_from_boxed: bool = True
    # only used if extract_from_boxed is False
    extract_regex: str = r"The final answer is (.+)$"
    # if True: try regex first, then boxed (regardless of extract_from_boxed)
    relaxed_extraction: bool = False
    take_modulo: int | None = None  # will take modulo of the gt and predicted answers if not None


@nested_dataclass(kw_only=True)
class LeanEvaluatorConfig(BaseEvaluatorConfig):
    sandbox: dict = field(default_factory=lambda: {"sandbox_type": "local"})
    num_parallel_requests: int = 10
    timeout: float = 30.0
    final_answer_key: str | None = None
    restate_formal_statement: bool = True
    # Which code block to extract when multiple are present: "first" or "last"
    extract_code_mode: str = "last"


class MathEvaluator(BaseEvaluator):
    def __init__(self, config: dict, num_parallel_requests=10):
        super().__init__(config, num_parallel_requests)
        self.eval_config = MathEvaluatorConfig(**self.config)

    async def eval_single(self, data_point: dict[str, any]) -> dict[str, any]:
        """Evaluate single problem for math"""
        if not self.eval_config.use_predicted_answer_key:
            data_point["predicted_answer"] = extract_answer(
                data_point["generation"],
                extract_from_boxed=self.eval_config.extract_from_boxed,
                extract_regex=self.eval_config.extract_regex,
                relaxed=self.eval_config.relaxed_extraction,
            )
        else:
            if "predicted_answer" not in data_point:
                raise ValueError(
                    "predicted_answer key not found in the data_point. Set use_predicted_answer_key=False to re-extract"
                )

        gt_answer = data_point["expected_answer"]
        predicted_answer = data_point["predicted_answer"]

        data_point["symbolic_correct"] = math_equal(
            gt_answer,
            predicted_answer,
            take_modulo=self.eval_config.take_modulo,
            numeric_precision=self.eval_config.numeric_precision,
            timeout_seconds=self.eval_config.timeout,
        )
        return data_point



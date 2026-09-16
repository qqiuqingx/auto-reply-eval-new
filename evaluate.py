#!/usr/bin/env python3
"""客服自动回复质量评估流水线（mock 模式）。"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_AUTO_REPLIES = PROJECT_ROOT / "data" / "task3_auto_replies.json"
DEFAULT_HUMAN_REF = PROJECT_ROOT / "data" / "task3_human_ref.json"
DEFAULT_MOCK_RESPONSES = PROJECT_ROOT / "data" / "mock_judge_responses.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts"

METRIC_DEFINITIONS = {
    "reference_consistency": {
        "name": "参考一致性",
        "weight": 0.25,
        "description": "与人工参考核心结论是否冲突，是否有内部矛盾或缺少依据的重要断言。",
    },
    "intent_completion": {
        "name": "核心诉求完成度",
        "weight": 0.40,
        "description": "是否直接回答并推进用户当前的核心问题，而非只给泛化知识。",
    },
    "actionability": {
        "name": "可执行性与服务主动性",
        "weight": 0.25,
        "description": "是否给出明确下一步、索取必要信息或主动查询处理。",
    },
    "tone": {
        "name": "场景化语气",
        "weight": 0.10,
        "description": "是否礼貌，并对投诉、焦虑、故障等场景作出恰当回应。",
    },
}

SCORE_MIN = 0
SCORE_MAX = 4
SEVERITIES = {"low", "medium", "high"}
PLACEHOLDER_RE = re.compile(r"(?<![A-Za-z0-9])XX(?![A-Za-z0-9])|<[^>]+>|\*{3,}")


class InputError(ValueError):
    """输入文件或评审返回不满足契约。"""


@dataclass(frozen=True)
class EvaluationCase:
    id: str
    user_question: str
    auto_reply: str
    human_reference: str
    annotator_notes: str
    reference_has_placeholder: bool


@dataclass(frozen=True)
class MetricResult:
    score: int
    reason: str


@dataclass(frozen=True)
class RiskFlag:
    code: str
    severity: str
    claim: str
    reason: str


@dataclass(frozen=True)
class JudgeResult:
    metrics: dict[str, MetricResult]
    missing_points: list[str]
    risk_flags: list[RiskFlag]
    summary: str


class JudgeClient(Protocol):
    mode: str

    def evaluate(self, case: EvaluationCase) -> JudgeResult:
        """返回一条符合评审输出契约的结果。"""


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise InputError(f"无法读取 JSON {path}: {exc}") from exc


def load_json_array(path: Path) -> list[dict[str, Any]]:
    payload = load_json(path)
    if not isinstance(payload, list):
        raise InputError(f"{path} 顶层必须是 JSON 数组")
    if not all(isinstance(item, dict) for item in payload):
        raise InputError(f"{path} 的每一项必须是对象")
    return payload


def require_non_empty_string(row: dict[str, Any], field: str, source: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise InputError(f"{source} 的字段 {field} 必须是非空字符串")
    return value


def build_cases(
    reply_rows: list[dict[str, Any]], reference_rows: list[dict[str, Any]]
) -> list[EvaluationCase]:
    replies_by_id: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(reply_rows, start=1):
        source = f"自动回复第 {index} 条"
        case_id = require_non_empty_string(row, "id", source)
        require_non_empty_string(row, "user_question", source)
        require_non_empty_string(row, "auto_reply", source)
        if case_id in replies_by_id:
            raise InputError(f"自动回复 id 重复: {case_id}")
        replies_by_id[case_id] = row

    refs_by_id: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(reference_rows, start=1):
        source = f"人工参考第 {index} 条"
        case_id = require_non_empty_string(row, "id", source)
        require_non_empty_string(row, "human_reference", source)
        notes = row.get("annotator_notes", "")
        if not isinstance(notes, str):
            raise InputError(f"{source} 的字段 annotator_notes 必须是字符串")
        if case_id in refs_by_id:
            raise InputError(f"人工参考 id 重复: {case_id}")
        refs_by_id[case_id] = row

    reply_ids = set(replies_by_id)
    ref_ids = set(refs_by_id)
    if reply_ids != ref_ids:
        raise InputError(
            "自动回复与人工参考 id 不一致；"
            f"缺失={sorted(reply_ids - ref_ids)}，多余={sorted(ref_ids - reply_ids)}"
        )

    cases: list[EvaluationCase] = []
    for row in reply_rows:
        ref = refs_by_id[row["id"]]
        reference = ref["human_reference"]
        cases.append(
            EvaluationCase(
                id=row["id"],
                user_question=row["user_question"],
                auto_reply=row["auto_reply"],
                human_reference=reference,
                annotator_notes=ref.get("annotator_notes", ""),
                reference_has_placeholder=bool(PLACEHOLDER_RE.search(reference)),
            )
        )
    return cases


def parse_metric(metric_key: str, payload: Any) -> MetricResult:
    if not isinstance(payload, dict):
        raise InputError(f"mock 指标 {metric_key} 必须是对象")
    score = payload.get("score")
    reason = payload.get("reason")
    if isinstance(score, bool) or not isinstance(score, int) or not SCORE_MIN <= score <= SCORE_MAX:
        raise InputError(f"mock 指标 {metric_key}.score 必须是 0～4 的整数")
    if not isinstance(reason, str) or not reason.strip():
        raise InputError(f"mock 指标 {metric_key}.reason 必须是非空字符串")
    return MetricResult(score=score, reason=reason)


def parse_risk_flag(index: int, payload: Any) -> RiskFlag:
    if not isinstance(payload, dict):
        raise InputError(f"mock risk_flags[{index}] 必须是对象")
    values: dict[str, str] = {}
    for field in ("code", "severity", "claim", "reason"):
        value = payload.get(field)
        if not isinstance(value, str) or not value.strip():
            raise InputError(f"mock risk_flags[{index}].{field} 必须是非空字符串")
        values[field] = value
    if values["severity"] not in SEVERITIES:
        raise InputError(f"mock risk_flags[{index}].severity 必须是 low/medium/high")
    return RiskFlag(**values)


def parse_judge_response(raw_response: str) -> JudgeResult:
    """解析 mock/真实模型共用的结构化返回契约。"""
    try:
        payload = json.loads(raw_response)
    except json.JSONDecodeError as exc:
        raise InputError(f"评审器返回的不是合法 JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise InputError("评审器返回顶层必须是对象")

    metric_payload = payload.get("metrics")
    if not isinstance(metric_payload, dict):
        raise InputError("评审器返回缺少 metrics 对象")
    expected = set(METRIC_DEFINITIONS)
    actual = set(metric_payload)
    if actual != expected:
        raise InputError(
            f"评审器指标集合不正确；缺失={sorted(expected - actual)}，多余={sorted(actual - expected)}"
        )
    metrics = {key: parse_metric(key, metric_payload[key]) for key in METRIC_DEFINITIONS}

    missing_points = payload.get("missing_points", [])
    if not isinstance(missing_points, list) or not all(
        isinstance(item, str) and item.strip() for item in missing_points
    ):
        raise InputError("评审器 missing_points 必须是非空字符串数组")

    risk_payload = payload.get("risk_flags", [])
    if not isinstance(risk_payload, list):
        raise InputError("评审器 risk_flags 必须是数组")
    risk_flags = [parse_risk_flag(index, item) for index, item in enumerate(risk_payload)]

    summary = payload.get("summary")
    if not isinstance(summary, str) or not summary.strip():
        raise InputError("评审器 summary 必须是非空字符串")
    return JudgeResult(metrics, missing_points, risk_flags, summary)


class MockJudgeClient:
    """以固定 JSON 响应模拟模型调用，不声称执行了真实语义推理。"""

    mode = "mock"

    def __init__(self, response_path: Path):
        payload = load_json(response_path)
        if not isinstance(payload, dict) or not all(
            isinstance(case_id, str) and isinstance(response, dict)
            for case_id, response in payload.items()
        ):
            raise InputError(f"{response_path} 顶层必须是 case id 到响应对象的映射")
        self._responses = payload
        self.response_path = response_path

    def evaluate(self, case: EvaluationCase) -> JudgeResult:
        payload = self._responses.get(case.id)
        if payload is None:
            raise InputError(f"mock 响应缺少 case: {case.id}")
        raw_response = json.dumps(payload, ensure_ascii=False)
        return parse_judge_response(raw_response)


def weighted_score(metrics: dict[str, MetricResult]) -> float:
    return round(
        sum(
            metrics[key].score / SCORE_MAX * definition["weight"] * 100
            for key, definition in METRIC_DEFINITIONS.items()
        ),
        1,
    )


def score_band(score: float) -> str:
    if score >= 85:
        return "优秀"
    if score >= 70:
        return "合格"
    if score >= 50:
        return "需要改进"
    return "不合格"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    if not results:
        raise InputError("自动回复数据不能为空")
    metric_summary: dict[str, Any] = {}
    for key, definition in METRIC_DEFINITIONS.items():
        values = [row["metrics"][key]["score"] for row in results]
        metric_summary[key] = {
            "name": definition["name"],
            "mean": round(sum(values) / len(values), 2),
            "distribution": dict(sorted(Counter(str(value) for value in values).items())),
        }
    risk_counts = Counter(
        flag["severity"] for row in results for flag in row["risk_flags"]
    )
    return {
        "case_count": len(results),
        "overall_mean": round(sum(row["overall_score"] for row in results) / len(results), 1),
        "band_distribution": dict(Counter(row["band"] for row in results)),
        "risk_flag_distribution": dict(risk_counts),
        "metrics": metric_summary,
        "worst_case_ids": [
            row["id"]
            for row in sorted(
                results,
                key=lambda item: (
                    item["overall_score"],
                    item["metrics"]["intent_completion"]["score"],
                    item["id"],
                ),
            )[:3]
        ],
    }


def render_report(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    result_by_id = {row["id"]: row for row in payload["results"]}
    lines = [
        "# 客服自动回复质量评估报告",
        "",
        f"- 样本数：{summary['case_count']}",
        f"- 评估模式：{payload['run']['mode']}（模拟返回，不代表真实模型判断）",
        f"- 整体平均分：**{summary['overall_mean']} / 100**",
        f"- 等级分布：{json.dumps(summary['band_distribution'], ensure_ascii=False)}",
        f"- 风险标记：{json.dumps(summary['risk_flag_distribution'], ensure_ascii=False)}",
        "",
        "## 指标结果",
        "",
        "| 指标 | 权重 | 均分（0-4） | 分布（分数:数量） |",
        "| --- | ---: | ---: | --- |",
    ]
    for key, definition in METRIC_DEFINITIONS.items():
        metric = summary["metrics"][key]
        distribution = "，".join(
            f"{score}:{count}" for score, count in metric["distribution"].items()
        )
        lines.append(
            f"| {definition['name']} | {definition['weight']:.0%} | "
            f"{metric['mean']} | {distribution} |"
        )

    lines.extend(
        [
            "",
            "## 逐条得分",
            "",
            "| Case | 综合分 | 等级 | 参考一致性 | 诉求完成度 | 主动性 | 语气 |",
            "| --- | ---: | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in payload["results"]:
        metrics = row["metrics"]
        lines.append(
            f"| {row['id']} | {row['overall_score']} | {row['band']} | "
            f"{metrics['reference_consistency']['score']} | "
            f"{metrics['intent_completion']['score']} | "
            f"{metrics['actionability']['score']} | {metrics['tone']['score']} |"
        )

    lines.extend(["", "## 最差 3 条", ""])
    for rank, case_id in enumerate(summary["worst_case_ids"], start=1):
        row = result_by_id[case_id]
        lines.extend(
            [
                f"### {rank}. {case_id} — {row['overall_score']} 分（{row['band']}）",
                "",
                f"- 用户问题：{row['user_question']}",
                f"- 自动回复：{row['auto_reply']}",
                f"- 评审摘要：{row['summary']}",
                "- 缺失要点：" + ("；".join(row["missing_points"]) or "无"),
                "- 风险："
                + (
                    "；".join(
                        f"[{flag['severity']}] {flag['reason']}" for flag in row["risk_flags"]
                    )
                    or "未标记"
                ),
                "",
            ]
        )

    placeholder_ids = [
        row["id"] for row in payload["results"] if row["reference_has_placeholder"]
    ]
    lines.extend(
        [
            "## 评估边界",
            "",
            "- 当前只有自动回复与人工参考，参考一致性不是绝对事实准确率。",
            "- mock 响应用于验证数据、评审、聚合和报告链路，不应用于上线决策。",
            "- 人工参考中的占位符仅从具体事实比对中排除，整条样本仍参与其他指标。",
            f"- 本次检测到占位符的参考样本：{', '.join(placeholder_ids) or '无'}。",
            "- 接入真实评审模型前，应补充政策、商品、订单和工具调用结果作为证据。",
            "",
        ]
    )
    return "\n".join(lines)


def evaluate_dataset(
    replies_path: Path,
    human_ref_path: Path,
    output_dir: Path,
    judge: JudgeClient,
) -> dict[str, Any]:
    cases = build_cases(load_json_array(replies_path), load_json_array(human_ref_path))
    if not cases:
        raise InputError("自动回复数据不能为空")

    results: list[dict[str, Any]] = []
    for case in cases:
        judged = judge.evaluate(case)
        overall = weighted_score(judged.metrics)
        results.append(
            {
                "id": case.id,
                "user_question": case.user_question,
                "auto_reply": case.auto_reply,
                "reference_has_placeholder": case.reference_has_placeholder,
                "metrics": {key: asdict(value) for key, value in judged.metrics.items()},
                "overall_score": overall,
                "band": score_band(overall),
                "missing_points": judged.missing_points,
                "risk_flags": [asdict(flag) for flag in judged.risk_flags],
                "summary": judged.summary,
            }
        )

    payload = {
        "run": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "mode": judge.mode,
            "disclaimer": "mock 结果是模拟评审返回，仅用于验证流水线。",
            "input": str(replies_path.resolve()),
            "input_sha256": sha256(replies_path),
            "human_reference": str(human_ref_path.resolve()),
            "human_reference_sha256": sha256(human_ref_path),
        },
        "metric_definitions": METRIC_DEFINITIONS,
        "summary": build_summary(results),
        "results": results,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "evaluation_results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "evaluation_report.md").write_text(
        render_report(payload), encoding="utf-8"
    )
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="客服自动回复质量评估流水线")
    parser.add_argument("--mode", choices=("mock",), default="mock", help="当前仅支持 mock")
    parser.add_argument("--auto-replies", type=Path, default=DEFAULT_AUTO_REPLIES)
    parser.add_argument("--human-ref", type=Path, default=DEFAULT_HUMAN_REF)
    parser.add_argument("--mock-responses", type=Path, default=DEFAULT_MOCK_RESPONSES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        judge = MockJudgeClient(args.mock_responses)
        payload = evaluate_dataset(args.auto_replies, args.human_ref, args.output_dir, judge)
    except InputError as exc:
        print(f"输入错误：{exc}")
        return 2
    summary = payload["summary"]
    print(
        f"完成 {summary['case_count']} 条 mock 评估；整体 {summary['overall_mean']}/100；"
        f"最差 3 条：{', '.join(summary['worst_case_ids'])}"
    )
    print(f"报告：{(args.output_dir / 'evaluation_report.md').resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

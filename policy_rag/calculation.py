"""Bounded Decimal arithmetic for model-proposed, evidence-linked expressions.

This calculates arithmetic; it does not decide policy eligibility, select rates or
validate a formula's business meaning. No eval, attributes, imports or tool access.
"""
from __future__ import annotations

import ast
from decimal import Decimal, DecimalException, ROUND_CEILING, ROUND_HALF_UP, localcontext
import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

_PLACEHOLDER = re.compile(r"\[\[calc:([a-z][a-z0-9_]{0,31})\]\]")
_NUMBER = re.compile(r"\d+(?:\.\d+)?")
_MAX_VALUE = Decimal("1000000000000")


class CalculationError(ValueError):
    pass


class CalculationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(pattern=r"^[a-z][a-z0-9_]{0,31}$")
    expression: str = Field(min_length=1, max_length=256)
    precision: int = Field(default=2, ge=0, le=6, strict=True)
    evidence_chunk_ids: list[str] = Field(min_length=1, max_length=8)


def evaluate_expression(expression: str) -> Decimal:
    """Evaluate a small arithmetic AST, preserving decimal literals exactly."""
    if not isinstance(expression, str) or not 1 <= len(expression) <= 256:
        raise CalculationError("Expression must contain 1 to 256 characters")
    try:
        tree = ast.parse(expression.strip(), mode="eval")
    except (SyntaxError, RecursionError) as exc:
        raise CalculationError("Invalid arithmetic syntax") from exc
    if sum(1 for _ in ast.walk(tree)) > 64:
        raise CalculationError("Expression is too complex")

    def checked(value: Decimal) -> Decimal:
        if not value.is_finite() or abs(value) > _MAX_VALUE:
            raise CalculationError("Arithmetic value exceeds supported range")
        return value

    def visit(node: ast.AST) -> Decimal:
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            literal = ast.get_source_segment(expression.strip(), node) or ""
            if not _NUMBER.fullmatch(literal):
                raise CalculationError("Use plain decimal numeric literals")
            return checked(Decimal(literal))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            operand = visit(node.operand)
            return checked(operand if isinstance(node.op, ast.UAdd) else -operand)
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
            left, right = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Add):
                value = left + right
            elif isinstance(node.op, ast.Sub):
                value = left - right
            elif isinstance(node.op, ast.Mult):
                value = left * right
            else:
                value = left / right
            return checked(value)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and not node.keywords:
            name = node.func.id
            if name not in {"ceil", "min", "max"} or not 1 <= len(node.args) <= 8:
                raise CalculationError("Only ceil, min and max are supported")
            args = [visit(arg) for arg in node.args]
            if name == "ceil":
                if len(args) != 1:
                    raise CalculationError("ceil takes one argument")
                return checked(args[0].to_integral_value(rounding=ROUND_CEILING))
            return checked(min(args) if name == "min" else max(args))
        raise CalculationError("Unsupported arithmetic operation")

    try:
        with localcontext() as context:
            context.prec = 28
            return visit(tree.body)
    except DecimalException as exc:
        raise CalculationError("Arithmetic failed, including division by zero") from exc


def resolve_calculations(
    answer: str, requests: list[CalculationRequest], evidence: list[Any],
) -> tuple[str, list[dict[str, Any]]]:
    """Resolve all requested placeholders or fail; never publish partial substitution."""
    if len(requests) > 6:
        raise CalculationError("At most six calculations are allowed")
    ids = [request.id for request in requests]
    placeholders = set(_PLACEHOLDER.findall(answer))
    if len(set(ids)) != len(ids) or set(ids) != placeholders:
        raise CalculationError("Calculation IDs must uniquely match answer placeholders")
    evidence_ids = {chunk.chunk_id for chunk in evidence}
    rendered, trace = {}, []
    for request in requests:
        if not set(request.evidence_chunk_ids) <= evidence_ids:
            raise CalculationError("Calculation refers to unavailable policy evidence")
        value = evaluate_expression(request.expression)
        displayed = value.quantize(Decimal(1).scaleb(-request.precision), rounding=ROUND_HALF_UP)
        rendered[request.id] = format(displayed, "f")
        trace.append({"id": request.id, "expression": request.expression, "value": str(value),
                      "rendered": rendered[request.id], "evidence_chunk_ids": request.evidence_chunk_ids})
    resolved = _PLACEHOLDER.sub(lambda match: rendered[match.group(1)], answer)
    if "[[calc:" in resolved:
        raise CalculationError("Invalid or unresolved calculation placeholder")
    return resolved, trace

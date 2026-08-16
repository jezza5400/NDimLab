import ast
import math
import operator
from collections.abc import Callable
from typing import ClassVar


class Evaluator:
	BINARY_OPERATORS: ClassVar[dict[type[ast.operator], Callable[[float, float], float]]] = {
		ast.Add: operator.add,
		ast.Sub: operator.sub,
		ast.Mult: operator.mul,
		ast.Div: operator.truediv,
	}

	UNARY_OPERATORS: ClassVar[dict[type[ast.unaryop], Callable[[float], float]]] = {
		ast.USub: operator.neg,
		ast.UAdd: operator.pos,
	}

	# Trig functions take/return degrees
	FUNCTIONS: ClassVar[dict[str, Callable[[float], float]]] = {
		"sin": lambda x: math.sin(math.radians(x)),
		"cos": lambda x: math.cos(math.radians(x)),
		"tan": lambda x: math.tan(math.radians(x)),
		"asin": lambda x: math.degrees(math.asin(x)),
		"acos": lambda x: math.degrees(math.acos(x)),
		"atan": lambda x: math.degrees(math.atan(x)),
		"sqrt": math.sqrt,
		"abs": abs,
		"exp": math.exp,
		"log": math.log,
	}

	@classmethod
	def evaluate_expression(cls, expr: str) -> float:
		"""Evaluates a math string safely adhering to BODMAS rules."""
		if not expr:
			return 0.0

		expr = expr.replace("·", "*")

		try:
			node = ast.parse(expr, mode="eval").body
			return cls._eval_node(node)
		except Exception as e:
			raise ValueError(f"Invalid math expression: '{expr}'") from e

	@classmethod
	def _eval_node(cls, node: ast.AST) -> float:
		if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
			return float(node.value)

		elif isinstance(node, ast.BinOp) and type(node.op) in cls.BINARY_OPERATORS:
			left = cls._eval_node(node.left)
			right = cls._eval_node(node.right)
			op_func = cls.BINARY_OPERATORS[type(node.op)]
			return op_func(left, right)

		elif isinstance(node, ast.UnaryOp) and type(node.op) in cls.UNARY_OPERATORS:
			operand = cls._eval_node(node.operand)
			op_func = cls.UNARY_OPERATORS[type(node.op)]
			return op_func(operand)

		elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id.lower() in cls.FUNCTIONS and len(node.args) == 1 and not node.keywords:
			arg = cls._eval_node(node.args[0])
			return cls.FUNCTIONS[node.func.id.lower()](arg)

		else:
			raise ValueError("Unsupported operation or expression structure")

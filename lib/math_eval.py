import ast
import math
import operator
from collections.abc import Callable
from typing import ClassVar


class Evaluator:
	"""
	Safe mathematical expression evaluator using Python's AST module.

	Supports:
	- Basic arithmetic (+, -, *, /)
	- Unary operators (+x, -x)
	- Trigonometric functions (sin, cos, tan, asin, acos, atan) in degrees
	- Common math functions (sqrt, abs, exp, log)

	Expressions are parsed using `ast.parse` to prevent unsafe execution.
	"""

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
		"""
		Evaluate a mathematical expression string safely using AST parsing.

		Supports arithmetic, unary operators, and functions defined in
		`Evaluator.FUNCTIONS`. Trigonometric functions operate in degrees.

		Args:
			expr (str): The expression to evaluate. Empty strings return 0.0.

		Returns:
			**result:** `float`
			The evaluated numeric result of the expression.

		Raises:
			ValueError: If the expression contains invalid or unsupported syntax.
		"""
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
		"""
		Recursively evaluate an AST node representing part of a math expression.

		Args:
			node (ast.AST): The AST node to evaluate.

		Returns:
			**value:** `float`
			The numeric result of evaluating the node.

		Raises:
			ValueError: If the node represents an unsupported operation.
		"""
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

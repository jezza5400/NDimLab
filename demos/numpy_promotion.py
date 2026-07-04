import numpy as np
from numpy.typing import NDArray


def vec_to_homogeneous(arr: NDArray) -> NDArray:
	# --- Case 1: single vector (n,) ---
	if arr.ndim == 1:
		n = arr.shape[0]
		H = np.eye(n + 1, dtype=arr.dtype)
		H[-1, :-1] = arr
		return H

	# --- Case 2: batch of vectors (m, n) ---
	elif arr.ndim == 2:
		m, n = arr.shape
		H = np.zeros((m, n + 1, n + 1), dtype=arr.dtype)

		# Fill identity blocks
		idx = np.arange(n + 1)
		H[:, idx, idx] = 1

		# Insert translation vectors
		H[:, -1, :-1] = arr

		return H

	else:
		raise ValueError("Input must be shape (n,) or (m,n)")


def linear_to_homogeneous(arr: NDArray) -> NDArray:
	# --- Case 1: single matrix (n, n) ---
	if arr.ndim == 2:
		n = arr.shape[0]
		hom = np.eye(n + 1, dtype=arr.dtype)
		hom[:n, :n] = arr
		return hom

	# --- Case 2: batch of matrices (m, n, n) ---
	elif arr.ndim == 3:
		m, n, n2 = arr.shape
		if n != n2:
			raise ValueError("Batch matrices must be square")

		hom = np.zeros((m, n + 1, n + 1), dtype=arr.dtype)
		idx = np.arange(n + 1)
		hom[:, idx, idx] = 1
		hom[:, :n, :n] = arr
		return hom

	else:
		raise ValueError("Input must be (n,n) or (m,n,n)")


arrs = np.array([[[1, 1], [2, 2]], [[4, 4], [5, 5]], [[10, 10], [20, 20]]])
vec_arrs = np.array([[1, 1], [2, 2], [5, 5]])
print(
	f"linear to homo:\n{arrs},\n\n--- Output ---\n\n{linear_to_homogeneous(arrs)}\n\nvec to homo:\n{vec_arrs},\n\n--- Output ---\n\n{vec_to_homogeneous(vec_arrs)}\n\n --- With in column vector form ---\n\n{vec_to_homogeneous(vec_arrs).swapaxes(1, 2)}"
)

import numpy as np
from numpy.typing import NDArray


def vec_to_homogeneous(arr: NDArray) -> NDArray:
	"""
	Convert a vector or batch of vectors into homogeneous transformation matrices.

	Supports:
	- Shape (n,) → returns a (n+1, n+1) homogeneous matrix
	- Shape (m, n) → returns a batch of (m, n+1, n+1) homogeneous matrices

	Args:
		arr (NDArray): Input vector or batch of vectors.

	Returns:
		**homogeneous:** `NDArray`
		A homogeneous matrix or batch of matrices embedding the input vector(s).

	Raises:
		ValueError: If the input array is not 1D or 2D.
	"""
	if arr.ndim == 1:
		n = arr.shape[0]
		h = np.eye(n + 1, dtype=arr.dtype)
		h[-1, :-1] = arr
		return h

	elif arr.ndim == 2:
		m, n = arr.shape
		h = np.zeros((m, n + 1, n + 1), dtype=arr.dtype)

		idx = np.arange(n + 1)
		h[:, idx, idx] = 1

		h[:, -1, :-1] = arr

		return h

	else:
		raise ValueError("Input must be shape (n,) or (m,n)")


def linear_to_homogeneous(arr: NDArray) -> NDArray:
	"""
	Convert a linear matrix or batch of linear matrices into homogeneous form.

	Supports:
	- Shape (n, n) → returns a (n+1, n+1) homogeneous matrix
	- Shape (m, n, n) → returns a batch of (m, n+1, n+1) homogeneous matrices

	Args:
		arr (NDArray): Square matrix or batch of square matrices.

	Returns:
		**homogeneous:** `NDArray`
		A homogeneous matrix or batch of matrices embedding the linear transform(s).

	Raises:
		ValueError: If the input is not square or not 2D/3D.
	"""
	if arr.ndim == 2:
		n = arr.shape[0]
		hom = np.eye(n + 1, dtype=arr.dtype)
		hom[:n, :n] = arr
		return hom

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

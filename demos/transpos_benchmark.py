import numpy as np
import timeit

view_once = 0
view_each = 0
view_copy = 0
make_view_cost = 0
make_copy_cost = 0
make_matrix_cost = 0

for _ in range(10):
	rotation = np.random.rand(1000, 1000)

	rotation_T = rotation.T
	rotation_T_copy = rotation.T.copy()

	t_view_once = timeit.timeit(lambda: rotation_T[500, 500], number=1_000_000) / 1_000_000
	t_view_each = timeit.timeit(lambda: rotation.T[500, 500], number=1_000_000) / 1_000_000
	t_copy = timeit.timeit(lambda: rotation_T_copy[500, 500], number=1_000_000) / 1_000_000

	# Measure cost of creating the matrix itself
	t_make_matrix = timeit.timeit(lambda: np.random.rand(1000, 1000), number=100) / 100

	# Measure cost of creating the .T view
	t_make_view = timeit.timeit(lambda: rotation.T, number=1_000_000) / 1_000_000

	# Measure cost of creating the .T.copy() full copy
	t_make_copy = timeit.timeit(lambda: rotation.T.copy(), number=100) / 100

	make_matrix_cost += t_make_matrix
	make_view_cost += t_make_view
	make_copy_cost += t_make_copy
	view_once += t_view_once
	view_each += t_view_each
	view_copy += t_copy

w = 30

print(f"{'Using stored view:':>{w}} {view_once / 10:.20f}")
print(f"{'Call .T each time:':>{w}} {view_each / 10:.20f}")
print(f"{'Full Copy:':>{w}} {view_copy / 10:.20f}")
print(f"{'Create matrix (1000x1000):':>{w}} {make_matrix_cost / 10:.20f}")
print(f"{'Create .T view:':>{w}} {make_view_cost / 10:.12f}")
print(f"{'Create .T.copy() full copy:':>{w}} {make_copy_cost / 10:.20f}")

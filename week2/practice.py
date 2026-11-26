"""
NumPy Practice: creating arrays, indexing, slicing, broadcasting, basic statistics.

"""

import numpy as np


def section(title):
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


# ----------------------------------------------------------------------
# 1. CREATING ARRAYS
# ----------------------------------------------------------------------
section("1. Creating arrays")

# EXERCISE 1.1: Create a 1-D array from the list [3, 1, 4, 1, 5, 9].
a = np.array([3, 1, 4, 1, 5, 9])
print("a =", a, "| dtype:", a.dtype, "| shape:", a.shape)
assert a.shape == (6,)

# EXERCISE 1.2: Create a 3x4 array of zeros and a 2x2 array of ones.
zeros = np.zeros((3, 4))
ones = np.ones((2, 2))
print("zeros:\n", zeros)
print("ones:\n", ones)
assert zeros.shape == (3, 4) and ones.sum() == 4

# EXERCISE 1.3: Make the integers 0..9 with arange, and 5 evenly spaced
# values from 0 to 1 (inclusive) with linspace.
r = np.arange(10)
ls = np.linspace(0, 1, 5)
print("arange:", r)
print("linspace:", ls)
assert np.allclose(ls, [0, 0.25, 0.5, 0.75, 1.0])

# EXERCISE 1.4: Create a 4x4 identity matrix and a 3x3 array filled with 7.
eye = np.eye(4)
sevens = np.full((3, 3), 7)
assert eye.trace() == 4 and (sevens == 7).all()

# EXERCISE 1.5: Create a 3x4 array containing 0..11 using reshape.
m = np.arange(12).reshape(3, 4)
print("m:\n", m)
assert m.shape == (3, 4)

# EXERCISE 1.6: Create reproducible random data: 5 uniform floats in [0, 1)
# and a 3x3 array of integers between 0 and 9 (seed = 42).
rng = np.random.default_rng(42)
uniform = rng.random(5)
rand_ints = rng.integers(0, 10, size=(3, 3))
print("uniform:", uniform)
print("rand_ints:\n", rand_ints)
assert uniform.shape == (5,) and rand_ints.max() <= 9


# ----------------------------------------------------------------------
# 2. INDEXING
# ----------------------------------------------------------------------
section("2. Indexing")

grid = np.array([[10, 11, 12, 13],
                 [20, 21, 22, 23],
                 [30, 31, 32, 33]])

# EXERCISE 2.1: Get the element in row 1, column 2 (expect 22).
el = grid[1, 2]
print("grid[1, 2] =", el)
assert el == 22

# EXERCISE 2.2: Get the last element using negative indices (expect 33).
last = grid[-1, -1]
assert last == 33

# EXERCISE 2.3: Use integer-array ("fancy") indexing to pull out the
# elements at (0,0), (1,2), (2,3) -> [10, 22, 33].
picked = grid[[0, 1, 2], [0, 2, 3]]
print("fancy:", picked)
assert (picked == [10, 22, 33]).all()

# EXERCISE 2.4: Boolean indexing: get all values greater than 21.
big = grid[grid > 21]
print("> 21:", big)
assert (big == [22, 23, 30, 31, 32, 33]).all()

# EXERCISE 2.5: Replace all values divisible by 10 with -1 (on a copy).
g2 = grid.copy()
g2[g2 % 10 == 0] = -1
print("g2:\n", g2)
assert g2[0, 0] == -1 and grid[0, 0] == 10  # original untouched

# EXERCISE 2.6: Use np.where to build an array that is "even"/"odd" labels
# for the first row.
labels = np.where(grid[0] % 2 == 0, "even", "odd")
print("labels:", labels)
assert (labels == ["even", "odd", "even", "odd"]).all()


# ----------------------------------------------------------------------
# 3. SLICING
# ----------------------------------------------------------------------
section("3. Slicing")

x = np.arange(20)

# EXERCISE 3.1: First 5 elements, last 5 elements, every 3rd element.
first5 = x[:5]
last5 = x[-5:]
every3 = x[::3]
print(first5, last5, every3)
assert (every3 == [0, 3, 6, 9, 12, 15, 18]).all()

# EXERCISE 3.2: Reverse the array.
rev = x[::-1]
assert rev[0] == 19 and rev[-1] == 0

# EXERCISE 3.3: From `grid`, get the first two rows and the last two columns.
sub = grid[:2, -2:]
print("sub:\n", sub)
assert (sub == [[12, 13], [22, 23]]).all()

# EXERCISE 3.4: Get the entire 2nd column (as 1-D) and the entire 1st row.
col1 = grid[:, 1]
row0 = grid[0, :]
assert (col1 == [11, 21, 31]).all() and (row0 == [10, 11, 12, 13]).all()

# EXERCISE 3.5: Slices are VIEWS, not copies. Prove it.
view = x[:5]
view[0] = 999
print("x[0] after modifying view:", x[0])
assert x[0] == 999
x[0] = 0  # restore

# ...and use .copy() when you want independence.
indep = x[:5].copy()
indep[0] = 999
assert x[0] == 0


# ----------------------------------------------------------------------
# 4. BROADCASTING
# ----------------------------------------------------------------------
section("4. Broadcasting")

# Rule of thumb: compare shapes from the RIGHT. Dimensions are compatible
# if they are equal or one of them is 1.

# EXERCISE 4.1: Add 100 to every element of `grid` (scalar broadcast).
plus = grid + 100
assert plus[0, 0] == 110

# EXERCISE 4.2: Add a row vector [1, 2, 3, 4] to every row of `grid`.
row_vec = np.array([1, 2, 3, 4])
res = grid + row_vec  # (3,4) + (4,) -> (3,4)
print("grid + row_vec:\n", res)
assert (res[0] == [11, 13, 15, 17]).all()

# EXERCISE 4.3: Add a column vector [[100], [200], [300]] to every column.
col_vec = np.array([[100], [200], [300]])
res = grid + col_vec  # (3,4) + (3,1) -> (3,4)
assert res[2, 0] == 330

# EXERCISE 4.4: Build a 5x5 multiplication table WITHOUT loops.
n = np.arange(1, 6)
table = n[:, None] * n[None, :]  # (5,1) * (1,5) -> (5,5)
print("multiplication table:\n", table)
assert table[4, 4] == 25 and table[2, 3] == 12

# EXERCISE 4.5: Pairwise distance matrix between 4 points on a line.
pts = np.array([0.0, 1.0, 3.0, 6.0])
dist = np.abs(pts[:, None] - pts[None, :])
print("distances:\n", dist)
assert dist[0, 3] == 6 and dist[1, 2] == 2

# EXERCISE 4.6: Normalize each COLUMN of a matrix to mean 0 (center it).
data = np.array([[1., 2., 3.],
                 [4., 5., 6.],
                 [7., 8., 9.]])
centered = data - data.mean(axis=0)  # (3,3) - (3,) -> (3,3)
print("centered:\n", centered)
assert np.allclose(centered.mean(axis=0), 0)

# EXERCISE 4.7 (think!): Why does this fail?  np.ones((3, 4)) + np.ones(3)
try:
    np.ones((3, 4)) + np.ones(3)
except ValueError as e:
    print("ValueError (expected):", e)
# Answer: shapes (3,4) and (3,) align from the right -> 4 vs 3, not
# compatible. Fix: np.ones(3)[:, None] makes it (3,1), which broadcasts.
fixed = np.ones((3, 4)) + np.ones(3)[:, None]
assert fixed.shape == (3, 4)


# ----------------------------------------------------------------------
# 5. BASIC STATISTICS
# ----------------------------------------------------------------------
section("5. Basic statistics")

scores = np.array([[82, 91, 77],
                   [65, 70, 88],
                   [90, 85, 95],
                   [58, 62, 71]])  # rows = students, cols = exams

# EXERCISE 5.1: Overall mean, median, std, min, max.
print("mean  :", scores.mean())
print("median:", np.median(scores))
print("std   :", scores.std())
print("min/max:", scores.min(), scores.max())
assert np.isclose(scores.mean(), 77.8333, atol=1e-3)

# EXERCISE 5.2: Mean score per exam (down the rows) and per student.
per_exam = scores.mean(axis=0)
per_student = scores.mean(axis=1)
print("per exam   :", per_exam)
print("per student:", per_student)
assert per_exam.shape == (3,) and per_student.shape == (4,)

# EXERCISE 5.3: Which student has the highest average? Which exam had the
# lowest average?  (argmax / argmin)
best_student = per_student.argmax()
hardest_exam = per_exam.argmin()
print("best student index:", best_student, "| hardest exam index:", hardest_exam)
assert best_student == 2

# EXERCISE 5.4: Percentiles and quartiles of all scores.
q1, q2, q3 = np.percentile(scores, [25, 50, 75])
print("quartiles:", q1, q2, q3)
assert q2 == np.median(scores)

# EXERCISE 5.5: Z-scores per exam (standardize columns) using broadcasting.
z = (scores - scores.mean(axis=0)) / scores.std(axis=0)
print("z-scores:\n", z.round(2))
assert np.allclose(z.mean(axis=0), 0) and np.allclose(z.std(axis=0), 1)

# EXERCISE 5.6: Cumulative sum and difference of the first student's scores.
cs = np.cumsum(scores[0])
diff = np.diff(scores[0])
print("cumsum:", cs, "| diff:", diff)
assert (cs == [82, 173, 250]).all()

# EXERCISE 5.7: Correlation between exam 1 and exam 2.
corr = np.corrcoef(scores[:, 0], scores[:, 1])[0, 1]
print("corr(exam1, exam2) =", round(corr, 3))
assert -1 <= corr <= 1

# EXERCISE 5.8: How many scores are >= 85, and what fraction is that?
mask = scores >= 85
print("count >= 85:", mask.sum(), "| fraction:", mask.mean().round(3))
assert mask.sum() == 5


# ----------------------------------------------------------------------
# 6. MINI CHALLENGE (combine everything)
# ----------------------------------------------------------------------
section("6. Mini challenge")

# Simulate 1000 dice rolls of two dice and estimate the probability that
# the sum is 7 (true value = 1/6 ≈ 0.1667).
rng = np.random.default_rng(0)
rolls = rng.integers(1, 7, size=(1000, 2))
sums = rolls.sum(axis=1)
p7 = (sums == 7).mean()
print("estimated P(sum = 7):", p7)
print("mean sum:", sums.mean(), "(expected 7.0)")
assert abs(p7 - 1 / 6) < 0.05


print("\nAll exercises passed!")
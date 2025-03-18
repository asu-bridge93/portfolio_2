N = int(input())
A = [["." for _ in range(N)] for _ in range(N)]

def kuro(A, i, j):
    for m in range(i, j + 1):
        for n in range(i, j + 1):
            A[m][n] = "#"
    return A

def siro(A, i, j):
    for m in range(i, j + 1):
        for n in range(i, j + 1):
            A[m][n] = "."
    return A

for i in range(N):
    j = N - i - 1
    if i <= j:
        if i % 2 == 0:
            A = kuro(A, i, j)
        else:
            A = siro(A, i, j)

for row in A:
    print("".join(row))

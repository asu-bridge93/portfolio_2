N = int(input())
A = list(map(int, input().split()))

left = set()
right = set()
l = 0
r = 0
l_list = []
r_list = []

for i in range(N):
    if A[i] not in left:
        l += 1
        left.add(A[i])
    l_list.append(l)

for i in range(N - 1, -1, -1):
    if A[i] not in right:
        r += 1
        right.add(A[i])
    r_list.append(r)

r_list.reverse()

m = 0
for i in range(N - 1):
    m = max(m, l_list[i] + r_list[i + 1])

print(m)

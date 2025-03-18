N = int(input())
A = list(map(int, input().split()))
l = 0
r = 1
def isDup(L):
    if len(L) == len(set(L)):
        return True
    else:
        return False
ans = []
while r <= N-1:
    if isDup(A[l:r]):
        r += 1
    else:
        ans.append(r-l)
        l += 1
if len(ans) == 0:
    print(-1)
else:
    print(min(ans))


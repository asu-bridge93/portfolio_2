Q = int(input())
card = [0]*100
for i in range(Q):
    A = list(map(int, input().split()))
    if A[0] == 1:
        card.append(A[1])
    else:
        print(card[-1])
        card = card[:-1]


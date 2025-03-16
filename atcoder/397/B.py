S = input()
n = len(S)

idx = 0
expected = 'i'
insertions = 0

while idx < n:
    if S[idx] == expected:
        idx += 1
    else:
        insertions += 1

    expected = 'o' if expected == 'i' else 'i'

if (n + insertions) % 2 == 1:
    insertions += 1

print(insertions)

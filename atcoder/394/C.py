def count_w_before_a(s: str) -> int:
    count_w = 0
    total = 0

    for char in s:
        if char == 'W':
            count_w += 1
        elif char == 'A':
            total += count_w
            count_w = 0
        elif char == 'C':
            count_w = 0
    
    return total

s = input().strip()
print(count_w_before_a(s))

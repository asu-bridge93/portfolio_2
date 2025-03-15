class Solution:
    def tribonacci(self, n: int) -> int:
        ans = [0, 1, 1]
        for i in range(40):
            ans.append(ans[-3] + ans[-2] + ans[-1])
        return ans[n]
        
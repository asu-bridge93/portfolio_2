class Solution:
    def findMaxAverage(self, nums: List[int], k: int) -> float:
        ans = [0] * (len(nums) + 1)
        for i in range(len(nums)):
            ans[i+1] = ans[i] + nums[i]
        l = []
        for i in range(len(nums)-k+1):
            l.append((ans[i+k] - ans[i])/k)
        return max(l)
        
        
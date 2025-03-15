class Solution:
    def kidsWithCandies(self, candies: List[int], extraCandies: int) -> List[bool]:
        ans = []
        m = max(candies)
        for i in range(len(candies)):
            ans.append(candies[i] + extraCandies >= m)
        return ans    
        
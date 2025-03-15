class Solution:
    def largestAltitude(self, gain: List[int]) -> int:
        alti = 0
        highest = 0
        for i in range(len(gain)):
            alti += gain[i]
            highest = max(alti, highest)
        return highest
        
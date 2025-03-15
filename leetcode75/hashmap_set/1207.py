class Solution:
    def uniqueOccurrences(self, arr: List[int]) -> bool:
        occ = {}
        for i in range(len(arr)):
            if arr[i] not in occ:
                occ[arr[i]] = 1
            else:
                occ[arr[i]] += 1
        return len(set(occ.values())) == len(occ)
        
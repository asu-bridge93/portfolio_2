class Solution:
    def reverseVowels(self, s: str) -> str:
        vowels = list('AIEOUaieou')
        pos = []
        for i in range(len(s)):
            if s[i] in vowels:
                pos.append(s[i])
        pos = pos[::-1]
        ans = ""
        a = 0
        for i in range(len(s)):
            if s[i] in vowels:
                ans += pos[a]
                a += 1
            else:
                ans += s[i]
        return ans
        
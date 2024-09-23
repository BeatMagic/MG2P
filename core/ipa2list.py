from typing import List, Tuple, Dict, Set
import sys
import os

tone_idx_to_pitch_contour = {
    "1": "˥",
    "2": "˧˥",
    "3": "˧˩˧",
    "4": "˥˩",
    "5": ""
}


class TrieNode:
    def __init__(self):
        self.children: Dict[str, 'TrieNode'] = {}
        self.is_end_of_word: bool = False


class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word: str):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end_of_word = True

    def search_from(self, s: str, start: int) -> List[str]:
        """
        从字符串 s 的位置 start 开始，使用 Trie 树查找所有可能的匹配 phoneme。
        返回所有匹配的 phoneme。
        """
        node = self.root
        matches = []
        current_phoneme = []
        for i in range(start, len(s)):
            char = s[i]
            if char in node.children:
                node = node.children[char]
                current_phoneme.append(char)
                if node.is_end_of_word:
                    matches.append(''.join(current_phoneme))
            else:
                break
        return matches


class PhonemeMatcher:
    def __init__(self, word_dict: Set[str]):
        """
        初始化 PhonemeMatcher，构建 Trie 树。

        :param word_dict: Set[str] - 包含所有 phoneme 的集合
        """
        self.trie = Trie()
        for word in word_dict:
            self.trie.insert(word)

    def tokenize(self, s: str) -> List[str]:
        """
        将输入的 xsampa 字符串拆分成 phoneme 序列，尽可能使用词表中的 phoneme，
        并在无法完全匹配时，选择编辑距离最小且 phoneme 数量最少的序列。

        :param s: str - 输入的 xsampa 字符串
        :return: List[str] - 输出的 phoneme 序列
        """
        n = len(s)
        # 初始化 DP 数组，dp[i] = (cost, phoneme_count, phone_list)
        dp: List[Tuple[int, int, List[str]]] = [(sys.maxsize, sys.maxsize, []) for _ in range(n + 1)]
        dp[0] = (0, 0, [])

        for i in range(n):
            current_cost, current_count, current_list = dp[i]
            if current_cost == sys.maxsize:
                continue  # 无法到达当前位置

            # 查找所有从位置 i 开始的匹配 phoneme
            matches = self.trie.search_from(s, i)

            if matches:
                for phoneme in matches:
                    end = i + len(phoneme)
                    new_cost = current_cost  # 匹配成功，无需增加编辑距离
                    new_count = current_count + 1
                    new_list = current_list + [phoneme]

                    if new_cost < dp[end][0]:
                        dp[end] = (new_cost, new_count, new_list)
                    elif new_cost == dp[end][0]:
                        if new_count < dp[end][1]:
                            dp[end] = (new_cost, new_count, new_list)
            else:
                # 没有匹配的 phoneme，考虑跳过当前字符，增加编辑距离
                new_cost = current_cost + 1
                end = i + 1
                new_count = current_count + 1  # 跳过一个字符也算作一个 phoneme
                new_list = current_list + [s[i]]

                if new_cost < dp[end][0]:
                    dp[end] = (new_cost, new_count, new_list)
                elif new_cost == dp[end][0]:
                    if new_count < dp[end][1]:
                        dp[end] = (new_cost, new_count, new_list)

        # 如果无法完全匹配，选择最优的近似匹配
        if dp[n][0] == sys.maxsize:
            # 找到所有可能的最小编辑距离
            min_cost = min(dp[i][0] for i in range(n + 1))
            # 选择最小编辑距离且 phoneme 数量最少的序列
            candidates = [dp[i] for i in range(n + 1) if dp[i][0] == min_cost]
            if candidates:
                # 选择 phoneme 数量最少的
                best = min(candidates, key=lambda x: x[1])
                return best[2]
            else:
                return []

        return dp[n][2]


def generate_639_3_map() -> dict:
    mapping = {}
    tsv_path = "MG2P/core/639_1to3.tsv"
    # tsv_path = "639_1to3.tsv"
    with open(tsv_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            language, prefix = parts[0], parts[1]
            mapping[language] = prefix
    return mapping


def load_ipa_dict():
    map639_3 = generate_639_3_map()
    ipa_dictionaries = {}
    matchers = {}
    for_unknown_lang_ipa_set = set()
    for language, prefix in map639_3.items():
        phone_tsv_path = 'MG2P/core/ipa_dict/'
        ipa_dict_path = os.path.join(phone_tsv_path, f'{prefix}.tsv')
        # check if the file exists
        if not os.path.exists(ipa_dict_path):
            continue
        all_ipa_set = set()
        with open(ipa_dict_path, 'r', encoding='utf-8') as f:
            for line in f:
                all_ipa_set.add(line.strip())
        for_unknown_lang_ipa_set |= all_ipa_set
        ipa_dictionaries[language] = all_ipa_set
        matchers[language] = PhonemeMatcher(all_ipa_set)
    matchers["unk"] = PhonemeMatcher(for_unknown_lang_ipa_set)
    ipa_dictionaries["unk"] = for_unknown_lang_ipa_set
    return matchers, ipa_dictionaries


def ipalist2phoneme(lyrics_ipa, matchers, tag) -> list:
    phone_matcher = matchers.get(tag, matchers["unk"])
    return [phoneme for ipa in lyrics_ipa for phoneme in phone_matcher.tokenize(ipa)]


if __name__ == '__main__':
    ipa_string = ['ˈtʃɑɹsiu', 'ˈɪs', 'ˈeɪ', 'ˈpɔɹk']
    matchers, ipa_dictionaries = load_ipa_dict()

    print(ipalist2phoneme(ipa_string, matchers, 'en'))

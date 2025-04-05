from itertools import combinations
from collections import defaultdict

class Apriori:
    def __init__(self, min_support=0.5):
        self.min_support = min_support
        self.freq_itemsets = []

    def fit(self, transactions):
        self.transactions = transactions
        self.transaction_count = len(transactions)
        self.item_support = defaultdict(int)

        # Step 1: Find frequent 1-itemsets
        itemsets = self.get_frequent_1_itemsets()
        k = 2

        while itemsets:
            self.freq_itemsets.extend(itemsets)
            # Generate candidates for k-itemsets
            candidates = self.generate_candidates(itemsets, k)
            itemsets = self.get_frequent_itemsets(candidates)
            k += 1

    def get_frequent_1_itemsets(self):
        item_counts = defaultdict(int)
        for transaction in self.transactions:
            for item in transaction:
                item_counts[frozenset([item])] += 1

        return [
            (itemset, count / self.transaction_count)
            for itemset, count in item_counts.items()
            if count / self.transaction_count >= self.min_support
        ]

    def generate_candidates(self, prev_freq_itemsets, k):
        prev_itemsets = [itemset for itemset, _ in prev_freq_itemsets]
        candidates = []

        for i in range(len(prev_itemsets)):
            for j in range(i+1, len(prev_itemsets)):
                union_set = prev_itemsets[i].union(prev_itemsets[j])
                if len(union_set) == k:
                    candidates.append(union_set)
        return list(set(candidates))  # Remove duplicates

    def get_frequent_itemsets(self, candidates):
        item_counts = defaultdict(int)

        for transaction in self.transactions:
            t_set = set(transaction)
            for candidate in candidates:
                if candidate.issubset(t_set):
                    item_counts[frozenset(candidate)] += 1

        return [
            (itemset, count / self.transaction_count)
            for itemset, count in item_counts.items()
            if count / self.transaction_count >= self.min_support
        ]

    def get_frequent_itemsets_result(self):
        return self.freq_itemsets
# Sample transactions
transactions = [
    ['milk', 'bread', 'butter'],
    ['beer', 'bread'],
    ['milk', 'bread', 'butter'],
    ['beer', 'bread', 'butter'],
    ['milk', 'bread'],
]

ap = Apriori(min_support=0.6)
ap.fit(transactions)

print("Frequent Itemsets:")
for itemset, support in ap.get_frequent_itemsets_result():
    print(f"{set(itemset)}: support = {support:.2f}")

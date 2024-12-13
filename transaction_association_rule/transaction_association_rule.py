import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from mlxtend.preprocessing import TransactionEncoder

data = pd.read_csv('transaction_dataset.csv')

data['items'] = data.groupby('Transaction')['Item'].transform(lambda x: ','.join(x))
data['items'] = data['items'].apply(lambda x: ','.join([item for item in x.split(',') if item != 'NONE']))

basket = data.groupby('Transaction')['items'].sum().reset_index(drop=True)

transactions = basket.map(lambda x: x.split(','))

encoder = TransactionEncoder()
encoded_data = encoder.fit(transactions).transform(transactions)

df_encoded = pd.DataFrame(encoded_data, columns=encoder.columns_)

min_support = 0.04
min_confidence = 0.6

frequent_itemsets = apriori(df_encoded, min_support=min_support, use_colnames=True)

rules = association_rules(frequent_itemsets, metric="lift", min_threshold=min_confidence, num_itemsets=len(basket))

print("Frequent Itemsets:")
print(frequent_itemsets)

print("\nAssociation Rules:")
print(rules)

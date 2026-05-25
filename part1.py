import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# ============================================================
# !! TODO - 팀 번호를 수정하세요 !!
# ============================================================
TEAM = 13

# ============================================================
# R1-1. 데이터 로드 및 horizontal table 생성
# ============================================================

# CSV 로드
df = pd.read_csv('DMA_project_UBR.csv')

# question을 index, tag을 column으로 하는 horizontal table 생성
# 각 셀: 해당 question에 해당 tag가 있으면 True, 없으면 False
df['value'] = True
horizontal = df.pivot_table(index='question', columns='tag', values='value', aggfunc='any')
horizontal = horizontal.fillna(False)

# ╔════════════════════════════════════════════════════════════╗
# ║  DO NOT MODIFY BELOW - R1-1 결과 저장                     ║
# ╚════════════════════════════════════════════════════════════╝
horizontal.to_pickle('DMA_project2_team%02d_part1_horizontal.pkl' % TEAM)
print(f"DMA_project2_team{TEAM:02d}_part1_horizontal.pkl 저장 완료")

# ============================================================
# R1-2. Frequent Itemsets & Association Rules
# ============================================================

# Frequent itemsets 생성 (min_support=0.005)
frequent_itemsets = apriori(horizontal, min_support=0.005, use_colnames=True)

# 연관 규칙 생성 (metric='lift', min_threshold=2.0)
rules = association_rules(frequent_itemsets, metric='lift', min_threshold=2.0)

# ╔════════════════════════════════════════════════════════════╗
# ║  DO NOT MODIFY BELOW - R1-2 결과 저장                     ║
# ╚════════════════════════════════════════════════════════════╝
rules.to_pickle('DMA_project2_team%02d_part1_association.pkl' % TEAM)
print(f"DMA_project2_team{TEAM:02d}_part1_association.pkl 저장 완료")

print("\nPart I 완료!")

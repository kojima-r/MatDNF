# 発現量データ
 
https://www.ebi.ac.uk/arrayexpress/experiments/E-MTAB-1908/
をRMA法で前処理をした発現量を利用

# File name

ファイル名：`<sample ID>_<expression type>.<data type>.csv`

- csvフォーマット（コンマ区切り）
- row：遺伝子（mRNA）
    各行の対応する遺伝子名はrow.attr.txtに記載
- column：時間（5分区切り）
    1列目0分,2列目5分,3列目10分,4列目15分,...


<sample ID> : sample ID = 01/02
このデータセットはサンプル２つのみ

<expression type>: T=total, L=labeled
発現量のうち修飾ありの発現量とトータルの発現量のデータがある

<data type> : nothing=original, std=standardized, bin=binary
処理内容：
<data type>なし=original：基本的な前処理のみ
std=standardized：標準化（ｚスコア化）
bin=binary：ｚスコアを０（平均）を閾値にして0/1化





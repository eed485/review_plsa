# review_plsa


# 研究概要
レビューデータから感性語と係り受け表現を抽出して，それらをモデル化します．

# 環境構築  
ライブラリの依存関係がややこしいので，新しく仮想環境を作ることをお勧めします．
> $ conda create -n plsa python=3.9.6  
> $ activate plsa

作った仮想環境内で必要なライブラリを入れます
> pip install ja-ginza spacy tqdm pandas

# 前処理
まずpreprocessing.pyから実行します．
## 準備
INPUT="input"
OUTPUT = "output"
でパスの入力フォルダと出力フォルダのパスの指定を行ってください
## 前処理
parse_documentでは係り受け分析の定義を行っています．
係り受け分析にはGINZAを用いています．GINZAはPOSタグというものを用いて係り受けの
関係を表すため，タグを指定する必要があります．
具体的には，
>["nsubj","obj","amod","acl","compound"]
の5種類のタグが付与される関係のものを定義しています．

（参考）https://qiita.com/kei_0324/items/400f639b2f185b39a0cf

その後，kakariukeで係り受け分析を実行しています．

次に感性語の抽出を行います．
感性語とは，印象語と感情語の両方を指す言葉で，「かわいい」「ほしい」などを指します.  
これを感性語の情報が記されているアプレイザル辞書というものを用い，それに合致するものを
レビュー文章から抽出します．  
tango_extractで名詞，形容詞，副詞，動詞を抽出します．  
次にkanseigo_extractで抽出された品詞の中で，アプレイザル辞書で感性語と定義されているものを
抽出しています．  
## 学習データの作成
本手法では，感性と係り受け表現をPLSAというトピックモデルを用いてモデル化をします．
PLSAでは，共起情報を用いてトピック抽出をすることが可能なため，共起行列を作成します．
kyouki_dfで共起情報を記したデータフレームを作成し，kyouki_df_to_matrixで共起行列にします.

# 学習
次にplsa.pyでPLSAを実行してください
## PLSAとは
### モデル
共起行列を基に感性と係り受け表現をモデル化していきます．
PLSAとはクラスタリング手法の1つで、  
1.文書dがP(d)で選ばれる  
2.トピックzがP(z|d)で選ばれる  
3.単語wがP(w|z)で生成される  
というモデルです。

僕の研究では
1.評価表現（形態要素＋外評価）dがP(d)で選ばれる  
2.印象トピックzがP(z|d)で選ばれる  
3.内評価wがP(w|z)で生成される  
という風にしています


```math
{P(d,w) = P(d)\sum_{z}P(z|d)P(w|z)
}
```
しかし，今回は  

```math
{P(d,w) = \sum_{z}P(z)P(d|z)P(w|z)
}
```
と式変形をして扱い，P(d)を評価表現(形態要素＋内評価)，P(w)を内評価とし，対数尤度
```math
{L = \sum_{d}\sum_{w}N(d,w)\log P(d,w)
}
```
が最大になる`P(z), P(d|z), P(w|z)`を、EMアルゴリズムを使って求めます．
N(d,w)は評価表現d内における内評価wの登場回数です。

### EMアルゴリズム
1.Estep
```math
{P(z|d,w) = \frac{P\left( z \right)P\left( d | z \right)P\left( w | z \right)}{\sum_{z} P\left( z \right)P\left( d | z \right)P\left( w | z \right)}
}
```
2.Mstep
```math
{\begin{eqnarray}
P\left( z \right) & = & \frac{\sum_{d} \sum_{w} N_{d, w} P\left( z | d, w \right)}{\sum_{d} \sum_{w} N_{d, w}} \\
P\left( d | z \right) & = & \frac{\sum_{w} N_{d, w} P \left( z | d, w \right)}{\sum_{d} \sum_{w} N_{d, w} P \left( z | d, w \right)} \\
P\left( w | z \right) & = & \frac{\sum_{d} N_{d, w} P \left( z | d, w \right)}{\sum_{d} \sum_{w} N_{d, w} P \left( z | d, w \right)}
\end{eqnarray}
```
対数尤度が収束するまで、E,Mstepを繰り返し計算しています．




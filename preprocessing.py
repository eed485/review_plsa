#必要なライブラリのインポート
import pandas as pd
import spacy
import ginza
from tqdm import tqdm
import json
import collections
import pickle
#ja-ginzaモデルのロード,文節区切りをCmodeに
nlp = spacy.load('ja_ginza')
ginza.set_split_mode(nlp, "C")

INPUT="input"
OUTPUT = "output"

#評価極性辞書の読み込み
f = open(INPUT + "/kanseigo_list.txt","rb")
alllist = pickle.load(f)
#データの読み込み
data = pd.read_csv(INPUT + '/kakaku_data.csv')
data = data[:30]
#係り受け分析の定義
def parse_document(sentence, nlp):
    doc = nlp(sentence)
    tokens = []

    for sent in doc.sents:
        for token in sent:
            tokens.append(token)

    subject_list = []

    for token in tokens:
        ## 依存関係ラベルが5つの関係のものをリストに追加する。
        if token.dep_ in  ["nsubj","obj","amod","acl","compound"]:
            #-感，ー性で抽出 ex)高級感，静粛性
            if token.text == "感":
                subject_list.append(f"{str(tokens[token.i-1].lemma_) + str(tokens[token.i].lemma_)}:{tokens[token.head.i].lemma_}")
            elif token.text == "性":
                subject_list.append(f"{str(tokens[token.i-1].lemma_) + str(tokens[token.i].lemma_)}:{tokens[token.head.i].lemma_}")
            
            else:
                subject_list.append(f"{tokens[token.i].lemma_}:{tokens[token.head.i].lemma_}")
    
    return subject_list

#係り受け分析の実行
def kakariuke(data):
    kakariuke_list = []
    print("-----------係り受け分析を開始-----------------------")
    for i in tqdm(range(len(data["0"]))):
        kakariuke = data["0"][i]
        kakariuke_list.append(parse_document(kakariuke,nlp))
    print("-----------係り受け分析を終了-----------------------")
    return kakariuke_list
#文章から名詞，形容詞，副詞，動詞抽出
def tango_extract(data):
    noun_toks = []
    noun_tok = []
    print("ーーーーーーー単語を抽出していますーーーーーーーーー")
    for i in tqdm(range(len(data))):
        kakariuke = "".join(data["0"][i])
        doc = nlp(kakariuke)
        for tok in doc:
            if tok.pos_ == 'NOUN':
                noun_tok.append(tok.text)
            elif tok.pos_ == 'PRON':
                noun_tok.append(tok.text)
            elif tok.pos_ == 'PROPN':
                noun_tok.append(tok.text)
            elif tok.pos_ == 'ADJ':
                noun_tok.append(tok.text)
        noun_toks.append(noun_tok)
        noun_tok = []
    print("ーーーーーーー終了しましたーーーーーーーーー")
    return noun_toks

#評価表現辞書に存在する印象語，感情語を抽出
def kanseigo_extract(noun_toks,alllist):
    inshou_kanjou = []
    tango_list = []
    print("----------感性語を抽出していますーーーーーーーーーーーーーーー")
    for count in tqdm(range(len(noun_toks))):
        for le in range(len(noun_toks[count])):
            if( (noun_toks[count][le] in alllist)== True):
                tango_list.append(noun_toks[count][le])
        inshou_kanjou.append(tango_list)  
        tango_list = []
    print("----------終了しましたーーーーーーーーーーーーーーー")
    return inshou_kanjou
#評価表現，感性語で共起行列をデータフレームで作成
def kyouki_df(inshou_kanjou,kakariuke_list):
    matrix = []
    for aa in tqdm(range(len(inshou_kanjou))):
        for k in range(len(inshou_kanjou[aa])):
            for j in range(len(kakariuke_list[aa])):
                matrix.append(inshou_kanjou[aa][k] +"+"+ kakariuke_list[aa][j])
                
    return matrix

#データフレームを二次元配列に変換
def kyouki_df_to_matrix(matrix):
    print("------------共起行列を作成します----------------")
    dic4 = collections.Counter(matrix)
    dic5 = sorted(dic4.items(), key=lambda x:x[1], reverse=True)
    mat = pd.concat([pd.DataFrame(dic5)[0].str.split('+', expand=True),pd.DataFrame(dic5)[1]],axis = 1)
    print("-------------作成完了しました-------------------")
    return mat


#↓↓↓↓↓↓↓↓↓↓↓↓前処理の実行↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓

#係り受け解析
kakariuke_list = kakariuke(data)
#感性語抽出
noun_toks = tango_extract(data)
inshou_kanjou = kanseigo_extract(noun_toks,alllist)
#共起行列の作成
matrix = kyouki_df(inshou_kanjou,kakariuke_list)
mat = kyouki_df_to_matrix(matrix)

mat.to_csv(OUTPUT + "/kyouki_matrix.csv")
print("実行完了です")



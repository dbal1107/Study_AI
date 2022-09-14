# 15.제로샷러닝_SBert모델사용_챗본구현.ipynb
# https://colab.research.google.com/drive/18sUw7kgN2EO9b3SGJ319Wwp8M_2xDll7

import pandas as pd
import pickle
from sentence_transformers import SentenceTransformer
import numpy as np


def df_save(fName, src_df):
  with open(fName, 'wb') as f:
    pickle.dump( src_df, f )

def df_load(fName):
  with open(fName, 'rb') as f:
    df = pickle.load( f )
  return df

model       = SentenceTransformer( 'xlm-r-100langs-bert-base-nli-stsb-mean-tokens' )
chatbot_df  = df_load( 'nlp/chatbot_df.dat' )

def cos_sim(AVec, BVec):
  return np.dot(AVec, BVec) / ( np.linalg.norm(AVec) * np.linalg.norm(BVec) )

def check_answer_similar( userSentence='' ):
  if not userSentence:
    return '정확하게 입력후 문의하세요'
  embeddingSentence   = model.encode( userSentence )  
  chatbot_df['score'] = chatbot_df.em.apply( lambda x: cos_sim( x, embeddingSentence) )
  return chatbot_df.loc[ chatbot_df['score'].idxmax() ]['A']

# -------------------------------------------------------
# 16.koBert를_이용한_이진분류.ipynb
# https://colab.research.google.com/drive/19RZrRDvcyrUvKPE8o0k28pa-79qnzyRj?usp=sharing

from transformers import TextClassificationPipeline
from transformers import TFBertForSequenceClassification
from transformers import BertTokenizerFast

# 가상환경에서 pip install tensorflow==2.8.2 설치
loadTokenizer = BertTokenizerFast.from_pretrained('./nlp/bert-base')
loadModel = TFBertForSequenceClassification.from_pretrained('./nlp/bert-base')
get_text_binary_clf = TextClassificationPipeline(
    tokenizer = loadTokenizer, # tokenizer 지정
    model = loadModel, # 모델 지정
    framework = 'tf', # 구동엔진 지정
    top_k = 1 # (팁) 유사도 기능 필요시 지정 (huggingface.co/models 참조)
)
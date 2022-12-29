# -*- coding: utf-8 -*-

from jobs import faiss_bert
import difflib,synonyms
import jieba.analyse
import jieba.posseg as posg
from word_similarity import WordSimilarity2010


def score_model(a1,a2,a3,data,query):
    result=[]
    for i, item in enumerate(data):
        s1=item.get("question")
        s2=item.get("text")
        cu=float(item.get("c"))
        score1=synonyms.compare(query,s1,seg=True)
        score2=synonyms.compare(query,s2,seg=True)
        print("c1={},c2={},id={}".format(score1,score2,item.get("_id")))
        score = a1*cu+a2*score1+a3*score2
        item["score"]=score
        result.append(item)
        result=sorted(result,key=lambda t: t.get("score"),reverse=True)
    return result

def score_model_div(a1,a2,a3,data,query):
    result=[]
    for i, item in enumerate(data):
        s1=item.get("question")
        s2=item.get("text")
        cu=float(item.get("c"))
        score1=difflib.SequenceMatcher(None,text_div(query,10),text_div(s1,10)).ratio()
        score2=difflib.SequenceMatcher(None,text_div(query,10),text_div(s2,10)).ratio()
        # print("c1={},c2={},id={}".format(score1,score2,item.get("_id")))
        score = a1*cu+a2*score1+a3*score2
        item["score"]=score
        result.append(item)
        result=sorted(result,key=lambda t: t.get("score"),reverse=True)
    return result

# 这个方法会给出用户list1[i]在目的词库list2[j]上的平均得分。
def word_sim_score(list1, list2):
    list =[]
    sum = 0.0
    # print(list1 + list2)
    ws = WordSimilarity2010()
    print("average:")
    for i,item1 in enumerate(list1):
        score_max = 0
        for i,item2 in enumerate(list2):
            #使用WordSimilarity
            # score = ws.similarity(item1,item2)
            #使用synonyms
            score = synonyms.compare(item1,item2,seg=False)

            print("{},{},{}".format(item1,item2,score))
            if score > score_max:
                score_max=score
        list.append(score_max)
    for i,num in enumerate(list):
        sum = sum + num
    return sum/len(list)

# 这个方法会给出用户list1[i]在目的词库list2[j]上的最大得分。
def word_sim_score_max(list1, list2):
    score_max_global = 0
    print("max:")
    for i,item1 in enumerate(list1):
        score_max_local = 0
        for i,item2 in enumerate(list2):

            score = synonyms.compare(item1,item2,seg=False)

            print("{},{},{}".format(item1,item2,score))
            if score > score_max_local:
                score_max_local=score

        if score_max_local > score_max_global:
            score_max_global = score_max_local
    return score_max_global

def score_model_wordsim(a1,a2,a3,data,query):
    result=[]
    for i, item in enumerate(data):
        s1=item.get("question")
        s2=item.get("text")
        cu=float(item.get("c"))
        score1=word_sim_score(text_div_list(query,10),text_div_list(s1,10))
        score2=word_sim_score(text_div_list(query,10),text_div_list(s2,10))
        print("c1={},c2={},id={}".format(score1,score2,item.get("_id")))
        score = a1*cu+a2*score1+a3*score2
        item["score"]=score
        result.append(item)
        result=sorted(result,key=lambda t: t.get("score"),reverse=True)
    return result
# 引入动词和名词的计算
def score_model_wordsim_nv(a1,a2,a3,data,query):
    result=[]
    for i, item in enumerate(data):
        s1=item.get("question")
        s2=item.get("text")
        s3=item.get("ns")
        s4=item.get("vs")
        cu=float(item.get("c"))
        score1=word_sim_score(text_div_list(query,10),text_div_list(s1,10))
        score2=word_sim_score(text_div_list(query,10),text_div_list(s2,10))

        sys_to_client= word_sim_score(ns_list(s3),text_div_list(query,10))
        client_to_sys= word_sim_score(text_div_list(query,10),ns_list(s3))
        if sys_to_client>client_to_sys:
            score3=sys_to_client
        else:
            score3 = client_to_sys
        score4= word_sim_score_max(text_div_vs(query,10),vs_list(s4))
        print("c1={},c2={},c3={},c4={},id={}".format(score1,score2,score3,score4,item.get("_id")))
        score = a1*cu+a2*score1+a3*score2
        item["score"]=score
        item["ns_score"]=score3
        item["vs_score"]=score4
        result.append(item)
        result=sorted(result,key=lambda t: t.get("score"),reverse=True)
    return result

# 为关键词的拼接
def text_div(str,count):
    taglist = jieba.analyse.extract_tags(str, topK=count, allowPOS=('n','vn','an','v','TIME','LOC','PER','nz','t'))
    print("taglist={}".format(taglist))
    return "".join(taglist)

# 分割关键词成列表
def text_div_list(str,count):
    # jieba.load_userdict("dict.txt")
    taglist = jieba.analyse.extract_tags(str, topK=count, allowPOS=('n','vn','an','v','TIME','LOC','PER','nz','t'))
    print("taglist={}".format(taglist))
    return taglist

# 关键词的动词列表
def text_div_vs(str,count):
    jieba.load_userdict("dict.txt")
    taglist = []
    words = posg.cut(str)
    for w in words:
        if w.flag == 'v':
            taglist.append(w.word)
        # print(w.word, w.flag)
    print("vs_taglist={}".format(taglist))
    return taglist

# 语料库中名词列表
def ns_list(str):
    list = str.split('&')
    print("ns_splitlist={}".format(list))
    if len(list)==0:
        return 0
    else:
        return list

# 语料库中动词列表
def vs_list(str):
    list = str.split('&')
    print("vs_splitlist={}".format(list))
    if len(list) == 0:
        return 0
    else:
        return list

def init():
    bc = faiss_bert.bert_client()
    # 第一次
    # index = faiss_bert.read_index_in_faiss("/opt/pythonworkspace/load_data/data_info")  # 读入index_file.index文件
    # 第二次
    index = faiss_bert.read_index_in_faiss("/opt/QASever/load_data/data_info_2.0_245")  # 读入index_file.index文件
    print("index_ntotal={}".format(index.ntotal))
    return bc, index

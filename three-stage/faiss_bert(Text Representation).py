# WX: aiwen2100

# bert-serving
from bert_serving.client import BertClient
import faiss

import numpy as np
from functools import reduce
import sys
import config.local_setting as config
sys.path.append('..')
from text2vec import SentenceModel,cos_sim,semantic_search, EncoderType

# dim = 384
dim = 768
# 初始化一个bertclient
def bert_client():
    bc = BertClient(config.BERT_SERVING_IP) # ip address of the GPU machine
    return bc

def sentence_bert_client():
    # bc = BertClient(ip='192.168.1.192')  # ip address of the GPU machine
    bc = SentenceModel("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",encoder_type=EncoderType.MEAN)
    return bc

def CoSENT_client():
    bc = SentenceModel("shibing624/text2vec-base-chinese",encoder_type=EncoderType.FIRST_LAST_AVG)
    return bc

# 向faiss索引库增加数据
def add_faiss(index,ids,datas):
    vectors = np.array( datas ).astype('float32') # vectors -> array
    ids = np.array(ids)
    #<ids,vectors>
    index.add_with_ids(vectors,ids)
# 归一化向量
def normaliz_vec(vec_list):
    for i in range(len(vec_list)):
        vec = vec_list[i]
        square_sum = reduce(lambda x, y: x+y, map(lambda x: x*x, vec))
        sqrt_square_sum = np.sqrt(square_sum)
        coef = 1/sqrt_square_sum
        vec = list(map(lambda x: x*coef, vec))
        vec_list[i] = vec
    return vec_list
# 将所有数据导入faiss中，并将faiss存储到磁盘index文件中
def add_in_faiss(task,bc,question_data_list):

    # 1. init faiss index
    index = faiss.index_factory(dim,"IDMap,Flat,L2norm",faiss.METRIC_INNER_PRODUCT)
    # index = faiss.index_factory(dim, "IDMap,Flat,L2norm", faiss.METRIC_L2)
    # index = faiss.index_factory(dim,"IDMap,Flat",faiss.METRIC_INNER_PRODUCT)
    # index = faiss.IndexHNSWFlat(dim,)

    batch_q_data = []
    batch_id_data = []

    for i,data in enumerate(question_data_list):
        id,question = data
        batch_q_data.append(question)
        batch_id_data.append(id)
        if i%100==0:
            print("create index (faiss) ,i = ",i)
            # 2. bert-serving ( text<batch_q_data> -> text vector)
            vectors = bc.encode( batch_q_data )
            vectors = normaliz_vec(vectors.tolist())
            add_faiss(index,batch_id_data,vectors) #index_db format:  <id,text vector> 
            batch_q_data.clear()
            batch_id_data.clear()

    # batch_q_data, batch_id_data -> bert-serving -> add data into faiss
    assert len(batch_q_data) == len(batch_id_data)
    vectors = bc.encode( batch_q_data )
    vectors = normaliz_vec(vectors.tolist())
    add_faiss(index,batch_id_data,vectors) #index_db format:  <id,text vector> 
    
    # 3. faiss index save
    table_name = "{}_{}".format(task,index.ntotal)
    faiss.write_index(index,"{}.index".format(table_name))
    print(index.ntotal ) # index total_count


# 将faiss加载至内存
def read_index_in_faiss():
    index = faiss.read_index(config.ABUSOLUTE_PATH + "common/index/" + config.INDEX_NAME)
    return index
# 使用query在index库中查询结果返回问题的相似度矩阵和整数编号索引
# topk指定返回前10条
def search_in_faiss(  bc,index,query,topK=10):

    # query = 兵马俑怎末走？

    # query -> bert-serving (text vector)
    query = [" ".join( [ word for word in query ])]
    print("bert-serving input: ",query)
    try:
        vectors = bc.encode( query )
        vectors = normaliz_vec(vectors.tolist())
    except:
        print('error: bert-serving disconnect,please check it')
        return [],[]
    # text vector (query)->  faiss ( return search list )

    print("-" * 60)
    try:
        query_list = np.array(vectors).astype('float32')
        dis,ind = index.search( query_list,k=topK )

        return dis,ind
    except:
        print("error: faiss service error")
        return [],[]
    return
    pass
from gensim.models import word2vec

'''
author = "kalafinaian"
email= "kalafinaian@outlook.com"
create_time = 2019-08-11
'''

'''
预料如何准备, 训练预料问津中每一行是一个文本，每个文本使用空进行分词
A B C ..
D E F ..
H I G ..
..
'''
s_corpus_url = "corpus.txt" # 语料库地址
sentences = word2vec.Text8Corpus(s_corpus_url,)  

'''
算法训练使用说明
架构：skip-gram（慢、对罕见字有利）vs CBOW（快）
训练算法：分层softmax（对罕见字有利）vs 负采样（对常见词和低纬向量有利）
欠采样频繁词：可以提高结果的准确性和速度（适用范围1e-3到1e-5）
文本（window）大小：skip-gram通常在10附近，CBOW通常在5附近
'''
train_model = word2vec.Word2Vec(sentences,
                        sg = 1,     # 0为CBOW  1为skip-gram
                        size = 300, # 特征向量的维度
                        window = 5, # 表示当前词与预测词在一个句子中的最大距离是多少
                        min_count = 5, # 词频少于min_count次数的单词会被
                        sample = 1e-3, # 高频词汇的随机降采样的配置阈值
                        iter = 23,  #训练的次数 
                        hs = 1,  #为 1 用hierarchical softmax   0 negative sampling
                        workers=8 # 开启线程个数
                        )

'''
模型的保存
'''
s_model_url = "train.model" # 语料库保存地址
train_model.save(s_model_url)

'''
模型的加载
'''
load_model = word2vec.Word2Vec.load(s_model_url)


'''
查询两个词的相似度
'''
s_word_1 = "关雎"
s_word_2 = "蒹葭"
f_word_sim = load_model.similarity(s_word_1, s_word_2)


'''
查询一个词的词向量, 返回是一个numpy数组
'''
s_query_word = ""
np_word  = load_model[s_query_word]


'''
打印一个词语所有相似词和相似度
'''
s_query_word = "华夏"
for s_word, f_sim in load_model.most_similar(s_query_word):
    print(s_word, f_sim)

'''
判断一个词语是否在词向量模型中
'''
s_word = "Naive"
if s_word in load_model.vocab:
    print("存在")
else:
    print("不存在")
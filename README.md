# text-classification
A text classification and similairty computing project in Python.We have tried wordbag,word2vec,WordMoverDistance,N-gram,LSTM,C-LSTM, LSTM with attention .etc.LSTM with attention(completed in Pytorch) turns out to be the best in out news title dataset.




## 摘要

​	本次作业的目标是实现新闻文本分类和寻找最相似新闻，我们在实现词袋子、词嵌入、词移距离等基本方法的基础上，尝试了N-gram、神经网络(LSTM和C-LSTM,以及带attention机制的版本)等方案，大幅提高了文本分类的准确度。下表是各种方案的准确率对比：

| 模型                                  | 准确率                  |
| ------------------------------------- | ----------------------- |
| Baseline（CountVectorizer建立词袋子） | 61.9%                   |
| TfidfVectorizer（maxfeatures=1000）   | 63.2%                   |
| TfidfVectorizer(维数不限)             | 82.9%                   |
| 词向量取平均                          | 67.9%                   |
| 序列相似度                            | 74%（前1000个测试数据） |
| Word Mover Distance                   | 72% （前1500个测试数据  |
| LSTM                                  | 84.3%                   |
| C-LSTM                                | 84.7%                   |
| LSTM_with_Attention                   | 85.9%                   |



以下为用词向量取平均和LSTM编码两种方案中找最相似文本的效果，都有很强的相关性:

![](https://res.cloudinary.com/dhyonw6zc/image/upload/v1557241812/56b7e294bf134f5f5cdcf081e399a7e.png)



![](https://res.cloudinary.com/dhyonw6zc/image/upload/v1557242032/43c25f6b79ddbdf17dcea14b2e54f7f.png)



以下为各种方案的核心算法概述和评测分析。

## 1.基础方法与改进

Baseline的做法是使用CountVectorizer建立词袋子作为句子的向量表示，准确率为61.9%,此处不再赘述。

考虑到只用词频不能反映词的重要程度，我们改用TF-IDF表征句向量，按照baseline设定max_features =1000时，准确率提升至63.2%，如果不设定max_features，句向量为七万多维，计算消耗时间较长，但准确率大幅度提升至82.9%。

~~~python
bowModel = TfidfVectorizer(stop_words=stopwords).fit(train_docs)
train_x = bowModel.transform(train_docs)
valid_x = bowModel.transform(valid_docs)
model = MultinomialNB() 
model.fit(train_x, train_labels)

prediction = model.predict(valid_x)
print('acc = %.4f' % (sum(prediction == valid_labels) / len(valid_labels)))

acc = 0.8293
~~~

以下是寻找最相似文本的代码与示例：

~~~python
def Find(query_str):
    seg = jieba.cut(query_str)
    vec = bowModel.transform([" ".join(seg)])
    score = np.zeros(train_x.shape[0])
    for i in range(train_x.shape[0]):
        diff = np.array(vec) - np.array(train_x[i])
        score[i] = abs(diff).sum()
    ids = list(range(train_x.shape[0]))
    ids.sort(key=lambda x:score[x])
    for i in ids[:20]:
        print(train_raw[i][0], train_raw[i][1])
    print("\n")

Find("王者荣耀国际版入选东南亚运动会电竞项目")
~~~

输出：

~~~
news_game 王者荣耀什么段坑？
news_game 怎么戒掉王者荣耀？
news_game 你为什么不玩王者荣耀了？
news_game 小孩子可以玩王者荣耀吗？
news_game 小孩子可以玩王者荣耀吗？
news_game 小孩子可以玩王者荣耀吗？
news_game 王者荣耀，吕布该怎么玩？
news_game 王者荣耀很火么？
news_game 你为什么卸载了王者荣耀？
news_game 王者荣耀怎么玩鲁班？
news_game 王者荣耀，谁的腿最长？
news_world 东南亚为什么叫东南亚，包括哪些国家？
news_game 马化腾玩王者荣耀吗？
news_game 你们喜欢玩王者荣耀还是吃鸡？
news_game 王者荣耀最难见到的皮肤有哪些，你们见到过吗？
news_game 王者农药那些坑！
news_game 段子手在荣耀的话
news_military 来自军体运动会上的“表情包”
news_game 王者荣耀：我擦这样都杀不死这小日本？你这暴君怕是假的吧？
news_game 《王者荣耀》怎么对付速推流？
~~~

输出的新闻虽然大多与王者荣耀相关，但没有体现出“电竞“,而且出现了“news_world 东南亚为什么叫东南亚，包括哪些国家？”这种不相干的新闻，有待提升。

考虑到词袋子不能体现出词义的相关关系，我们用Word Embedding优化计算。

Word2vec是Google于2013年的Distributed Representations of Words and Phrases and their Compositionality 以及后续的Distributed Representations of Words and Phrases and their Compositionality 两篇文章中提出的一种高效训练词向量的模型, 基本出发点是上下文相似的两个词,它们的词向量也应该相似。

![](http://jalammar.github.io/images/word2vec/word2vec.png)



Word2vec需要用到神经网络计算，我们利用gensim库的word2vec将词表示为300维的向量，训练代码如下(后续的LSTM等方案中使用的也是此处训练的词向量)。

~~~python
import gensim 
self_train_model = gensim.models.Word2Vec([doc.split() for doc in train_docs]+[doc.split() for doc in valid_docs] ,min_count=1,sg=1,size=300,iter=50)    
self_train_model.wv.save_word2vec_format('train_result.txt')
self_train_model.most_similar("北大") 
~~~

~~~
[('校长', 0.6423088312149048),
 ('林建华', 0.5577366352081299),
 ('上读', 0.5569656491279602),
 ('女硕士', 0.554108738899231),
 ('错读', 0.5495955944061279),
 ('校庆', 0.5492573380470276),
 ('洪浩', 0.5448577404022217),
 ('鸿鹄', 0.5442037582397461),
 ('师叔你', 0.5393882393836975),
 ('错字', 0.5357639789581299)]
~~~

直接对词向量取平均表示句子，分类时准确率为68%左右，比max_features=1000时的词袋子模型有提升：

~~~python
import pandas as pd 
new_train_cv,new_valid_cv = [],[]
for sentence in train_docs:
    sentence = sentence.split()
    vec = np.zeros(300)
    for word in sentence:  
        if word in self_train_model.wv.vocab :    
            vec += self_train_model[word]*1/len(sentence)
        else:
            vec += np.zeros(300)
    #print(vec)
    new_train_cv.append(vec)
    
new_valid_cv = []
for sentence in valid_docs:
    sentence = sentence.split()
    vec = np.zeros(300)
    for word in sentence:
        if word in self_train_model.wv.vocab:     
            vec += self_train_model[word]*1/len(sentence)
        else:
            vec += np.zeros(300)
    #print(vec)
    new_valid_cv.append(vec)
    
newmodel = GaussianNB()
newmodel.fit(new_train_cv, train_labels) 

prediction = newmodel.predict(new_valid_cv)    
print('acc = %.4f' % (sum(prediction == valid_labels) / len(valid_labels))) 
~~~

~~~
acc = 0.6787
~~~

寻找最相似新闻即寻找离自身距离最近的句向量对应的文本，效果比起词袋子有很大改善，出现了电竞相关的内容，而且没有游戏之外其他类的消息：

~~~python
def Find2(query_str):
    seg = jieba.cut(query_str)
    seg = " ".join(seg).split()
    vec = np.zeros(300)
    for word in seg:
        if word in self_train_model.wv.vocab:   
            vec += self_train_model[word]/len(seg)
        else:
            vec += np.zeros(300)
    score = np.zeros(len(new_train_cv))
    for i in range(len(new_train_cv)): 
        diff = np.array(vec) - np.array(new_train_cv[i])
        score[i] = abs(diff).sum()
    ids = list(range(len(new_train_cv)))    
    ids.sort(key=lambda x:score[x])
    for i in ids[:20]: 
        print(train_raw[i][0], train_raw[i][1])     
    print("\n")   

Find2("王者荣耀国际版入选东南亚运动会电竞项目") 
~~~

~~~
news_game 王者荣耀：一局巅峰赛三个国服，国服第一杨戬寒冰称霸巅峰赛！
news_game 王者荣耀S11赛季五黑阵容推荐 花木兰有望回归T0上单？
news_game 王者荣耀：国服第一之间的电竞盛宴！KPL明星赛梦泪老帅再次联手
news_game 王者荣耀：玩家晒图开黑节狂上21星成荣耀王者，你升段了吗？
news_game 电竞，继世界杯和NBA后，第三个全民竞技
news_game 王者荣耀之大陆起源、各国势力分布！绝对可以拍成一部史诗级电影
news_game 《王者荣耀》巅峰赛中，阿珂BAN率已达到100%，能否说明阿珂已成为最强打野？
news_game 《王者荣耀》巅峰赛中，阿珂BAN率已达到100%，能否说明阿珂已成为最强打野？
news_game 《王者荣耀》巅峰赛中，阿珂BAN率已达到100%，能否说明阿珂已成为最强打野？
news_game Faker入选韩国电竞名人堂，如果中国也有，哪些人能入选
news_game 王者荣耀巅峰赛，前5名中都有哪些大神？
news_game 王者荣耀：所有国服第一都在这里 五五开黑节梦泪老帅再度联手？
news_game 王者荣耀：玩家5000元买下王者大号，登入游戏后乐坏了！
news_game 王者荣耀：巅峰赛阿轲惹了众怒，成必BAN英雄，已成为最强打野？
news_game 王者荣耀：四位强势中单，干将莫邪排名第一，嬴政只能屈局第四！
news_game 王者荣耀：体验服5.3更新，多位英雄调整
news_game 王者荣耀S11赛季什么英雄是打到高端局一定要会的？
news_game 世界上最大的格斗游戏比赛EVO，街霸5报名人数仅排第二
news_game 王者荣耀各个英雄的国服第一分别是谁？
news_game 真·国服第一诺手问世？余小C韩服王者600点诺手豪取五杀！
~~~

至此，我们还没有考虑词序的因素，第二部分将尝试序列相似度、词移距离、N元词组等方案优化相似度计算。

## 2.利用序列相似度/词移距离/N-gram优化相似度计算

### 2.1 序列相似度方案

------

考虑词序关系，将句子中所有词转成索引，看成一个序列。利用Levenshtein距离（编辑距离）计算两个序列的相似度。

即定义句子A与B的编辑距离为：只进行如下3种操作，将A 变成B的最少操作数。


>* 在任意位置增加一个词

>* 删除任意一个词

>* 将一个词替换成另一个词



具体求解使用动态规划。设求序列$A$与序列$B$的编辑距离，$f[i][j]$为序列$A$前$i$个词与序列$B$前$j$个词的编辑距离。

$$
f[i][j]=\left\{
\begin{aligned}
&f[i][j-1]+1\\
&f[i-1][j]+1\\
&f[i-1][j-1]+cost
\end{aligned}
\right.
$$
其中$$ cost=\left\{
\begin{aligned}
0, A[i-1]&==B[j-1]\\
1, A[i-1]&!=B[j-1]
\end{aligned}
\right.
$$
最后$ f[len(A)][len(B)] $即为$A,B$的编辑距离。由于各个句子长度不同，我们将$\frac{f[len(A)][len(B)]}{len(A)+len(B)}$作为最后的相似度。
实验中观测到，将替换操作的代价设为2似乎更好。

具体实现如下：
        
    def levenshtein_dis(x, y):
    len_str1 = len(x) + 1
    len_str2 = len(y) + 1
    #create matrix
    matrix = [0 for n in range(len_str1 * len_str2)]
    #init x axis
    for i in range(len_str1):
        matrix[i] = i
    #init y axis
    for j in range(0, len(matrix), len_str1):
        matrix[j] = j 
          
    for i in range(1, len_str1):
        for j in range(1, len_str2):
            if x[i-1] == y[j-1]:
                cost = 0
            else:
                cost = 2
            matrix[j*len_str1+i] = min(matrix[(j-1)*len_str1+i]+1,
                                        matrix[j*len_str1+(i-1)]+1,
                                        matrix[(j-1)*len_str1+(i-1)] + cost)
    return matrix[-1] / float(len_str1+len_str2-2)

因为样本过大，采用KNN算法进行分类复杂度较高，未能测试全部数据。对1000个数据进行测试，得到正确性约为$74\%$。

查找相似文本任务中，表现差强人意。
如
Find("首只独角兽周二上市！8个涨停赚4万")
与其最相似的20个标题为：

------
news_finance    首只独角兽周二上市！8个涨停赚4万
news_finance 一盆冷水，首只港股独角兽平安好医生，上市首日差点破发！
news_finance 独角兽药明康德周二上市，但一季报净利已现业绩下滑
news_finance 首只独角兽药明康德，还未上市，就宣布一季度净利润同比下滑！
news_car 传祺GM8上市直接卖18万
news_tech 如何在互联网上一个月赚2万？
news_finance 投资一千，赚1041万的心酸
news_house 从事什么行业可以半年内赚60万？
news_finance 有30万本金，怎么在短时间内赚到100万？
news_finance 猪粪生产有机肥，万吨赚68万
news_finance 巴菲特卖酒赚了7000万；华山论剑西凤酒提价
news_entertainment 范丞丞一张自拍一夜赚480万，你如何看这件事？
news_entertainment 范丞丞是谁？还不是靠姐姐范冰冰名气，有什么资格赚学生480万？
news_finance 如何选择涨停的个股？许多散户不一定知道的，多学一分钟都是赚！
news_finance 5年赚了500万，原来出租车司机是个理财高手！
news_finance 一年怎么赚63万 巴菲特告诉你10大法宝
news_finance 炒股十年了，我得到了什么？一个月赚8万也买不回青春
news_tech 靠捡垃圾2年赚了1000万，这才是真的帅！
news_finance 叶斯喻：千三不破，下周看涨，把握行情，再赚100万！
news_finance 地摊谁说不能年赚30万，看人家怎么做的……

------

其中大致可分为两个类型：独角兽上市和赚*万
这显然是由编辑距离的计算方法导致的，这一方法只单纯的考虑了句子结构，而没有考虑到语义。
那么是否可以加入语义作为参考呢？

考虑将句子中的词转为词向量，然后计算编辑距离保留替换操作，代价变为$cost=1-cos\_sim(A[i-1],B[j-1])$其中$cos\_sim$为计算余弦相似度。


实验结果：
Find("首只独角兽周二上市！8个涨停赚4万")

-----
news_finance 首只独角兽周二上市！8个涨停赚4万
news_finance 独角兽药明康德周二上市，但一季报净利已现业绩下滑
news_finance 药明康德今日上市，逾40万手买单封涨停
news_finance 首只独角兽药明康德，还未上市，就宣布一季度净利润同比下滑！
news_sports 张路足彩逆天10连中！魔球理论神预测 轻松赚245万
news_car 这15个新能源车标，认识8个就算铁杆车迷！
news_car 吉利全新中级车即将上市，外观竟比奥迪A7还漂亮
news_world 卖给本国人300，卖给中国人1万
news_finance 沙河股份：业绩狂涨！一季报业绩增193.51%！
news_tech 小米7价格首曝！屏幕指纹6GB售2799元
news_house 小学合并后房价要变！某项目退款首付达3000万
news_finance 农民工回乡创业 抓住这4个项目 抢先赚到第一桶金
news_tech 你们的小米要上市了，还不快来看看！
news_house 上海 4 月份新房供应量创 22 个月来新高！
news_house 高周转为王 前五房企4个月销售额逼近万亿
news_game 季中赛首日uzi对线欧成！排位赛开始练这个英雄！
news_house 政府卖地收入暴涨83.4%！这几个城市正在悄然崛起
news_house 政府卖地收入暴涨83.4%！这几个城市正在悄然崛起
news_finance 10只个股KDJ即将金叉，短线极具爆发力！
news_house 杭州摇号最低价来了！微风之城领出11500元！

-----

可以发现，效果没达到预期，原因可能是替换操作的代价没有设好，没平衡好语义和结构。

### 2.2 词移距离方案

Word2Vec将词映射为一个词向量，在这个向量空间中，语义相似的词之间距离会比较小，而词移距离（WMD）正是基于word2vec的这一特性开发出来的。两篇文章的词移距离是：![](http://nooverfit.com/wp/wp-content/uploads/2017/02/Screenshot-from-2017-02-02-100311.png)

这里的$x_i$其实是被word2vec压缩过的d维的隐藏层词向量.

$D(x_i,x_j)$代表两篇文章最短距离的实际意义是: **所有文档a中单词转移到文档b中单词的最短总距离**. 换句话说, 两篇文档总距离中的每个单词距离分两部分计算:一是单词间的word2vec距离；二是**单词xi放到另一篇文档中应该转换为单词xj的权重:*。优化的目标是使总和最小，类似于网络流算法。

![](http://nooverfit.com/wp/wp-content/uploads/2017/01/Screenshot-from-2017-01-31-124438.png)

我们利用词移距离寻找最相似的文本，为了节省计算时间，只在训练集的一部分进行寻找(这里随机取10000个)，为了节省计算时间，并取neighbors = 20进行简单的KNN分类，返回所属的类别。

~~~python
def new_find(query_str):
    seg = "".join(jieba.cut(query_str)) 
    new_train_docs = train_docs 
    new_train_labels = train_labels 
    new_train_data = list(zip(new_train_docs,new_train_labels)) 
    random.shuffle(new_train_data)
    #new_train_data[:1000]  
    new_train_docs = []
    new_train_labels = []
    for item in new_train_data[:10000]:
        item = list(item)
        new_train_docs.append(item[0])
        new_train_labels.append(item[1]) 
    score = np.zeros(len(new_train_docs))
    for i in range(len(new_train_docs)): 
        score[i] = model.wmdistance(new_train_docs[i],seg)
    ids = list(range(len(new_train_docs)))  
    ids.sort(key=lambda x:score[x])
    lis = [0]*12
    for i in ids[:20]:   
        print("".join(new_train_docs[i].split()))   
        lis[new_train_labels[i]] += 1 
    ans = 0 
    for i in range(0,12):
        if lis[i] > lis[ans]: 
            ans = i 
    print(ans)    
    #print("\n")
    return ans 

new_find("王者荣耀国际版入选东南亚运动会电竞项目")
~~~

输出:

~~~
王者荣耀各个英雄的国服第一分别是谁？
王者荣耀里你最希望得到的绝版皮肤是哪个？
王者荣耀：对面选这些英雄，千万别拖后期
王者荣耀多久没登陆才会送snk英雄？
运动会都有什么项目？
《王者荣耀》哪个英雄到后期最垃圾？
《王者荣耀》会不会倒闭？
王者荣耀一些知名主播为什么不选择加入职业战队？
王者荣耀：这三个无视版本的英雄，因为太强都没人玩？
你在王者荣耀遇到过的最奇葩名字是什么？
王者荣耀如果双方都不打，只让小兵上，谁会赢？
玩过王者荣耀后，去玩LOL是什么体验？
你有多久没有打开王者荣耀了？
王者荣耀：五个国服孙尚香就这伤害，天美：你在逗我
王者荣耀：你的亚瑟还在当坦克玩？换套出装体会秒人的快感
王者荣耀：练好这几个英雄，不上王者算我输！
《王者荣耀》中你最喜欢的女英雄是谁？
王者荣耀有哪些再不加强就没法玩了的英雄？
王者荣耀里面哪个段位最恐怖？
你们现在玩王者荣耀是什么感觉？
~~~

遍历测试集验证准确度，计算耗时较大，已知计算至1500个样本时准确率为72% 

~~~python
acc = 0 
for i,doc in enumerate(valid_docs):
    if new_find(doc) == valid_labels[i]:
        acc += 1 
    print(str(i+1)+"/"+str(18000)+" acc = " +str(acc/(i+1)) ) 

~~~

### 2.3 N-gram方案

#### 工作概述

- 优化了2.2中的Levenstein计算模型，增加其运行速度，使后面尝试KNN聚类变为可行
- 参考引入了BLEU中对句子N-Gram的考量，扩展了原有的句子相似度模型
- 对数据进行了最相似句子查找和KNN聚类的任务，并且评估其结果和N-Gram优化的效果

#### 核心算法概述

- **Levenshtein距离**：
  又称编辑距离，指的是将字符串看做序列，两个字符串之间，由一个转换成另一个所需的最少编辑操作次数。许可的编辑操作包括将一个字符替换成另一个字符，插入一个字符，删除一个字符。是计算字符串相似度的常用算法。本项目中使用python-Levenshtein包中基于C语言的相关函数库，较之其他方法速度更快

- **N-Gram相似度模型**：
  大致思路是将两字符串分别看做1-gram，2-gram直到N-gram的序列，分别计算其Levenshtein距离，最后求加权平均。理论上由于其评价句子中词组的相似度，n-gram相似度高比uni-gram相似度高更能说明句子主题相近。由于所使用包只支持计算字符串的距离，故建立了一个词数组到Unicode字符串的哈希函数，计算哈希得到的字符串的相似度的KNN算法。
- **KNN聚类方法**采用无监督的模型的机器学习快速高效得到接过。由于数据集valid set太大，在测试时使用训练集的全部和valid set的部分（每一类采样20个，共采样12类）用时测试了分词时有没有停用词和knn邻居数量分别是5和15的情况，和n-gram中n<=1,2,3等情况。

#### 评测与分析

- **停用词与n-gram**： 由于n-gram主要考察句子中的连续因素，所以理论上只有在没去停用词的句子上才会有效。在原先不去掉停用词之后，算法最容易找到的就是虚词的相似，找到句型相似的而不是主题相似的。可以在sim文件中看到不同n最同一个句子计算的影响

***Given:美记：奥尔特曼计划和胡德会面 将根据会面情况做出处罚(N = 1):***

Sentence:巴菲特：美中关系可以实现双赢 将扩大在华投资
Distance:11.0

------

Sentence:上联：暖日和风送春雨，如何对下联？
Distance:12.0

------

Sentence:古风：衣袂翩翩 沐雪而来
Distance:12.0

------

Sentence:上联：徽宗擅书画 请对下联
Distance:12.0

------

Sentence:上联：暖日和风送春雨，如何对下联？
Distance:12.0

------

***Given:美记：奥尔特曼计划和胡德会面 将根据会面情况做出处罚(N = 2):***

Sentence:巴菲特：美中关系可以实现双赢 将扩大在华投资
Distance:11.5

------

Sentence:玄德品金：5.6投资黄金原油 将是你盈利的开始
Distance:12.0

------

Sentence:北京限价房销售新规征求意见 将是谁的利好
Distance:12.0

------

Sentence:保时捷新款Macan动力升级 将于今年8月亮相
Distance:12.0

------

Sentence:雀巢拟收购星巴克零售咖啡业务 将支付71.5亿美元先期付款
Distance:12.0

------

***Given:美记：奥尔特曼计划和胡德会面 将根据会面情况做出处罚(N = 3):***

Sentence:巴菲特：美中关系可以实现双赢 将扩大在华投资
Distance:11.666666666666666

------

Sentence:玄德品金：5.6投资黄金原油 将是你盈利的开始
Distance:12.0

------

Sentence:北京限价房销售新规征求意见 将是谁的利好
Distance:12.0

------

Sentence:保时捷新款Macan动力升级 将于今年8月亮相
Distance:12.0

------

Sentence:雀巢拟收购星巴克零售咖啡业务 将支付71.5亿美元先期付款
Distance:12.0

- **聚类结果**：neighbour数为15的均由于neighbour为5的，由于去了停用词之后很多标题长度只有4，所以最多测试到n=3。在240大小的测试集和全部训练集上进行，ARI分数如下：
  - 有stop词：0.023239525265594863，0.046118858020052264，0.04858496933369259（n = 1,2,3）
  - 没有：0.05221258478219645，0.05528967464517933，0.05259511605617368（n = 1,2,3)
  - 可见有n=2时有没有停词结果都是最好的，去掉停词准确率会上升。

## 3.神经网络部分

### 数据预处理
- 第一步，对于training set和valid set进行数据清洗，首先进行分词，再从中剔除空的字符串、标点符号以及所有的stopword，再从中筛选出只含汉字或者英文字母的词，构建出vocabulary。  
- 第二步：使用word2vec进行embedding，embedding size=300

### 模型结构
- 采用双层、单向的LSTM，在最后一层之后添加一个线性层和softmax层进行分类。
```python
    def forward(self, x):
        # 将词语的下标转换成word embedding
        e_x = self.embedding(x)

        # 初始hidden_state为None
        r_out, (h_n, h_c) = self.rnn(e_x, None)

        # 取最后时刻的hidden_state再通过一个全连接层输出
        # 这里不用加softmax, pytorch的CrossEntropyLoss函数已经封装好了log-softmax
        out = self.out(r_out[:, -1, :])
        return out
```
- 模型参数：
    - embedding_size=300(w2v) or 768(BERT)
    - hidden_size=128(w2v) or 512(BERT)
    - output_size=12（标题有12类）

### 训练细节
采用Adam，初始学习率设置成0.01，batch_size=64，每一个epoch结束后，学习率按照5%的比率衰减，也即变为原来的95%。训练3个epoch即可。

### 训练结果
在valid set上，最高的accuracy可以达到84.3167%，以下是训练时每100个batch输出的loss和accuracy信息（不是全部）：
```code
Epoch:  1  | train loss: 0.6283  | test accuracy: 0.840778
Epoch:  1  | train loss: 0.3840  | test accuracy: 0.830944
Epoch:  1  | train loss: 0.2300  | test accuracy: 0.841333
Epoch:  1  | train loss: 0.3020  | test accuracy: 0.837278
Epoch:  1  | train loss: 0.5474  | test accuracy: 0.835000
Epoch:  1  | train loss: 0.7205  | test accuracy: 0.841000
Epoch:  2  | train loss: 0.1891  | test accuracy: 0.842167
Epoch:  2  | train loss: 0.2643  | test accuracy: 0.837056
Epoch:  2  | train loss: 0.2075  | test accuracy: 0.842000
Epoch:  2  | train loss: 0.1628  | test accuracy: 0.840556
Epoch:  2  | train loss: 0.1551  | test accuracy: 0.834667
Epoch:  2  | train loss: 0.3220  | test accuracy: 0.838778
Epoch:  2  | train loss: 0.1875  | test accuracy: 0.843167
Epoch:  2  | train loss: 0.3354  | test accuracy: 0.842444
Epoch:  2  | train loss: 0.2906  | test accuracy: 0.837944
Epoch:  2  | train loss: 0.5087  | test accuracy: 0.841833
Epoch:  2  | train loss: 0.2058  | test accuracy: 0.842500
Epoch:  2  | train loss: 0.2439  | test accuracy: 0.842611
```

### 模型优化

综合卷积神经网络和循环神经网络（即C-LSTM模型），在LSTM前加入卷积层：

~~~python
self.conv = nn.ModuleList([nn.Conv2d(1, kernel_num, (K,input_size)) for K in kernel_sizes]) 
~~~

可以将accuracy提高到84.7167%。

~~~
Epoch:  0  | train loss: 0.5067  | test accuracy: 0.833056
Epoch:  0  | train loss: 0.4047  | test accuracy: 0.839000
Epoch:  1  | train loss: 0.4418  | test accuracy: 0.832778
Epoch:  1  | train loss: 0.3545  | test accuracy: 0.840222
Epoch:  1  | train loss: 0.2390  | test accuracy: 0.847167
Epoch:  1  | train loss: 0.1727  | test accuracy: 0.836667
Epoch:  1  | train loss: 0.6123  | test accuracy: 0.840500
~~~

在原本模型的基础上增加self-attention机制，将词向量改成预训练好的BERT词向量，最高的accuracy可以达到85.8889%

```code
Epoch:  1  | train loss: 0.1474  | test accuracy: 0.854833
Epoch:  1  | train loss: 0.3972  | test accuracy: 0.853111
Epoch:  1  | train loss: 0.4958  | test accuracy: 0.858889
Epoch:  1  | train loss: 0.4836  | test accuracy: 0.855389
Epoch:  2  | train loss: 0.2277  | test accuracy: 0.858778
```

### 结果分析

我们选取了一个表现较好的模型，输出其预测错误的部分，我们选取一些典型的例子：

| 序号 | 句子                                                     | predict       | truth              |
| ---- | -------------------------------------------------------- | ------------- | ------------------ |
| 0    | 贝克汉姆大儿子携女友当街秀恩爱，准儿媳像公主小七长相甜美 | news_sports   | news_entertainment |
| 1    | LOL笑笑离婚事件内情曝光后：将辞去MSI季中赛解说一职！     | news_game     | news_entertainment |
| 2    | 再叙“一个真实的故事”                                     | news_military | news_entertainment |
| 3    | 一句“在吗？”，钱就没有了？是什么让人闻“在吗”色变？       | news_culture  | news_car           |
首先可以看到，有一些新闻标题有一定的“跨界”性，比如贝克汉姆这个例子，我们的模型准确地预测出了贝克汉姆和sports的相关性，但对于后面”秀恩爱“等词的侧重不够，没有把握到这句话的重点。而笑笑这则新闻，划分为game这一类别实际上也没有太大的问题。还有一些新闻从标题上本就无法准确知道它是属于哪一类的，这给分类工作也带来了一定的困难（例如句子2和3）。  
解决这个问题，一个思路是直接去掉干扰词，例如：将句子0中的主语“贝克汉姆”去掉，再输入网络，得到的结果是entertainment，是准确的，但这样的方法不一定总是奏效，确定什么样的词是干扰词，很依赖于句法分析。利用哈工大提供的dependency parsing工具，我们可以对句法进行分析。
```python
def Get_dependency(sentence):
    return HanLP.parseDependency(sentence)
```

### 寻找最相似

我们使用RNN中最后一个hidden state当做句子的向量，再以向量之间的曼哈顿距离计算句子相似度，找到最相似的20个句子。代码和效果如下：

~~~python
model = torch.load('model/w2v_nonstop_512_0.95decay_attn[2].pkl',map_location='cpu')
word2idx = load_data('word2idx.pkl')
label2id = {'news_culture': 0,
            'news_entertainment': 1,
            'news_sports': 2,
            'news_finance': 3,
            'news_house': 4,
            'news_car': 5,
            'news_edu': 6,
            'news_tech': 7,
            'news_military': 8,
            'news_travel': 9,
            'news_world': 10,
            'news_game': 11}
def Word2idx(word):
    try:
        return word2idx[word]
    except:
        return word2idx['Unknown']
        
def Get_sentence_type_and_embedding(sentence):
    word_list = list(jieba.cut(sentence))
    word_index = [Word2idx(word) for word in word_list]
    #print(word_index)
    out1, out2 = model(Variable(torch.LongTensor([word_index])))
    index = torch.squeeze(out1).max(0)[1].item()
    return list(label2id.keys())[index], torch.squeeze(out2).cpu().data.numpy()

# 获取所有句子的编码    
train_cv = [] 
for i,doc in enumerate(train_docs): 
    t,v = Get_sentence_type_and_embedding(doc) 
    train_cv.append(v)  
    if i % 100 == 0 :
        print(i) 
        

def Find_most_similar(sentence):
    type,vect = Get_sentence_type_and_embedding(sentence) 
    score = np.zeros(len(train_docs)) 
    for i,doc in enumerate(train_docs):
        v = train_cv[i]   
        diff = np.array(vect) - np.array(v) 
        score[i] = abs(diff).sum() 
        
    ids = list(range(len(train_cv)))    
    ids.sort(key=lambda x:score[x])
    for i in ids[:20]: 
        print(train_raw[i][0], train_raw[i][1])     
    print("\n")   

# 寻找最相似的句子

Find_most_similar("王者荣耀国际版入选东南亚运动会电竞项目") 
~~~

输出

~~~
news_game GMB战队成为第一支从英雄联盟MSI入围赛晋级的队伍，怎么评价这支战队的表现？
news_game 外卡战队全部晋级失败！FW轻松3-0吊打“和KZ五五开”的GMB!
news_game 打游戏时，你有哪些惊为天人的神操作？
news_game 王者战士排行榜，首位居然是最冷门的英雄！
news_game 季中赛：老M5三连胜 土耳其豪门打破最短比赛纪录
news_game 网络游戏时代，你还喜欢玩“单机游戏”吗？
news_game 打游戏的时候，你见过最没素质的人是什么样的？
news_game 玩家最讨厌这一点，拳头花了8年时间都没办法解决！
news_game 玩家求官方加强猴子，别改李白，策划一一答复
news_game 小智三年坐稳全民一哥，而她成为一姐只用了三个月！
news_game 网络游戏精美图集 游戏原画欣赏 那些不为人知的作品
news_game CSGO战队世界最新排名 中国TyLoo战队创历史最佳排名！
news_game 韦神反向Q被LOL官方翻出来恶搞，那一箭，飞了三年终究是回来了！
news_game 玩家为什么在Steam打差评？这些人统计了6224款游戏得到了答案
news_game 季中赛入围赛A组收官 GMB5胜1负晋级下轮
news_game LPL裁判小姐姐微博曝光，RNG粉丝称幸运女神终于找到你了！
news_game LPL裁判小姐姐微博找到了，这届网友太差了，五一都过了才找到
news_game 打游戏爱骂人的人，是什么心理？
news_game 打游戏爱骂人的人，是什么心理？
news_game 打游戏爱骂人的人，是什么心理？
~~~

靠前的内容都是和王者荣耀的电竞比赛相关的内容，较之前方案的效果相关性更强。

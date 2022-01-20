# Retrieval-based Neural Source Code Summarization
code for ICSE 2020 paper "Retrieval-based Neural Source Code Summarization"

### 参考视频

https://www.bilibili.com/video/BV1S34y1i7AW



### 要求
* Hardwares: 16 cores of 2.4GHz CPU, 128GB RAM and a Titan Xp GPU with 12GB memory (GPU is a must )
* OS: Ubuntu 16.04/ Windows
* PS：因为Python版本和pytorch版本的改变，我修改了部分代码以保证程序能够照常运行。
* Packages:
	+ python 3.9 (for runing the main code)
	+ pytorch 1.10.1
	+ torchtext 0.3.1
	+ nltk 3.2.4
	+ ConfigArgParse 0.14.0

### 第一步：数据格式处理
1、在samples目录下新建文件夹，例如我们跑的是bash，就新建bash文件夹

2、分别创建train/test/valid/preprocessed/output/indexes 文件夹，用来后续的数据存放

3、将训练集的代码保存在train目录下，命名为train.code.src，将代码对应的ast序列命名为train.ast.src，将代码对应的注释命名为train.nl.tgt。
测试集和验证集同样的操作。

### 第二步：对数据进行预处理
1、修改run.py中的相关代码

```python
if __name__ == '__main__':
    ......
    assert lang in ['python', 'java', 'bash']
    if lang == 'python':
        src_len, tgt_len = 100, 50
    elif lang == 'java':
        src_len, tgt_len = 300, 30
    elif lang == 'bash':
        src_len, tgt_len = 30, 30
    else:
        print("Unsupported Programming Language:", lang)
```

以bash为例，在断言中加入‘’bash‘’，在对应的输入最大长度和输出最大长度修改为30，30（根据数据集的长度分布进行选择）

2、命令行中运行python run.py preprocess bash，vocabulary结果保存在preprocessed目录下。

### 第三步：训练神经网络模型

1、命令行运行python run.py train bash，参数修改都在onmt/opts.py文件里

2、找到Validation accuracy最高的模型，记录下来。这里我们跑的是baseline_spl_step_10000.pt

### 第四步：检索

1、因为安装pylucene太麻烦了，我们进行了修改，改成了textdistance库中的textdistance.levenshtein.normalized_similarity 方法。

2、修改run.py中的命令行，最重要的是使用的模型参数，我们跑的是baseline_spl_step_10000.pt

```python
elif opt == 'retrieval':
    print('Syntactic level...')
    command1 = "python syntax_new.py %s" % lang
    os.system(command1)
    print('Semantic level...')
    batch_size = 32 if lang == 'python' else 16
    command2 = "python translate.py -model models/%s/baseline_spl_step_10000.pt \
                    -src samples/%s/train/train.code.src \
                    -output samples/%s/output/test.out \
                    -batch_size %d \
                    -gpu 0 \
                    -fast \
                    -max_sent_length %d \
                    -refer 0 \
                    -lang %s \
                    -search 2" % (lang, lang, lang, batch_size, src_len, lang)
    os.system(command2)
    command3 = "python translate.py -model models/%s/baseline_spl_step_10000.pt \
                    -src samples/%s/test/test.code.src \
                    -output samples/%s/test/test.ref.src.1 \
                    -batch_size 32 \
                    -gpu 0 \
                    -fast \
                    -max_sent_length %d \
                    -refer 0 \
                    -lang %s \
                    -search 2" % (lang, lang, lang, src_len, lang)
    os.system(command3)
    print('Normalize...')
    command4 = "python normalize.py %s" % lang
    os.system(command4)
```

3、命令行运行python run.py retrieval bash

### 第五步：混合方法生成注释

1、修改run.py中的命令行代码，包括模型参数，输出的最小长度，beam size等等

```python
elif opt == 'translate':
    command = "python translate.py -model models/%s/baseline_spl_step_10000.pt \
                -src samples/%s/test/test.code.src \
                -output samples/%s/output/Rencos.out \
                -min_length 3 \
                -max_length %d \
                -batch_size 32 \
                -gpu 0 \
                -fast \
                -max_sent_length %d \
                -refer %d \
                -lang %s \
                -beam 5" % (lang, lang, lang, tgt_len, src_len, mode, lang)
    os.system(command)
    print('Done.')
```

2、命令行运行python run.py translate bash 2，结果输出在samples/bash/output/Rencos.out

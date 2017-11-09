#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Shared by http://www.tensorflownews.com/
# Github:https://github.com/TensorFlowNews
import jieba
from tensorflow.contrib import learn
import numpy as np

str_neg='''结巴中文分词'''
seg_list = jieba.cut(str_neg, cut_all=False)
word_list=[item for item in seg_list]
print(word_list)
word_str=' '.join(word_list)
print(word_str)
max_document_length = len(word_list)
vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
x = np.array(list(vocab_processor.fit_transform([word_str])))

print(vocab_processor.vocabulary_)
print(vocab_processor)


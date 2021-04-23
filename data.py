import os
import csv
import json
import torch
import logging
import math
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from itertools import chain
from transformers import GPT2Tokenizer
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__file__)
logging.basicConfig(level=logging.INFO) 



	
	
def pad_to_length(tokens, pad, block_size, test=False):
	length = len(tokens)
	if test:
		tokens += (block_size-length)*[pad]
	else:
		tokens += ((math.ceil(length/block_size) - math.floor(length/block_size))*block_size - length%block_size)*[pad]
	return tokens, length



def create_dataloader_train(text, pad, batch_size, args): 
	chunks = []
	lm_labels = []
	padded_text, ln = pad_to_length(text, pad, args.block_size)
	ln_filled = len(padded_text)
	for i in range(0, ln_filled, args.block_size):
		tokens = padded_text[i:i+args.block_size]
		labels = [i if i!=pad else -100 for i in tokens]
		chunks.append(tokens)
		lm_labels.append(labels)
	token_ids = torch.tensor(chunks)
	lm_labels = torch.tensor(lm_labels)
	dataloader = DataLoader(TensorDataset(token_ids,lm_labels), batch_size)
	return dataloader
	

	
def create_dataloader_test(texts, tokenizer, args): 
	examples, lm_labels, lengths = [], [], []

	tokenized_text = texts.essay.apply(lambda x: tokenizer.convert_tokens_to_ids(tokenizer.tokenize(x))).values
	max_length = max([len(i) for i in tokenized_text])
	length = (math.ceil(max_length/args.block_size))*args.block_size
	batch_size = length//args.block_size

	for test_item in tokenized_text:
		padded_text, ln = pad_to_length(test_item, tokenizer.pad_token_id, length, True)
		for i in range(0, length, args.block_size):
			tokens = padded_text[i:i+args.block_size]
			labels = [i if i!=tokenizer.pad_token_id else -100 for i in tokens]
			examples.append(tokens)
			lm_labels.append(labels)
			lengths.append(ln)
	
	token_ids = torch.tensor(examples)
	lm_labels = torch.tensor(lm_labels)
	lengths = torch.tensor(lengths)

	dataloader = DataLoader(TensorDataset(token_ids,lm_labels,lengths), batch_size)
	return dataloader
	
	

	
	
def create_dataloader_folds(args, tokenizer, lang, splits, texts, labs):
	print('Creating Folds for %s...' %lang)
	folds, labels = [], []
	for train_index, test_index in splits:
		train_texts, test_texts = texts.loc[train_index,:], texts.loc[test_index,:]
		train_labels, test_labels = labs.loc[train_index,:], labs.loc[test_index,:]
		train_texts = " ".join(train_texts[train_labels.label==lang].essay.values)
		tokenized_train_texts = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(train_texts))

		train_loader =  create_dataloader_train(tokenized_train_texts, tokenizer.pad_token_id, args.train_batch_size, args)
		test_loader = create_dataloader_test(test_texts, tokenizer, args)
		folds.append([train_loader, test_loader])
		labels.append(test_labels.label.values)
	return folds, labels




def create_dataloader(args, tokenizer, lang, texts, labs, test_texts):
	print('Creating data for %s...' %lang)
	train_texts = " ".join(texts[labs.label==lang].essay.values)
	tokenized_train_texts = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(train_texts))

	train_loader =  create_dataloader_train(tokenized_train_texts, tokenizer.pad_token_id, args.train_batch_size, args)
	test_loader = create_dataloader_test(test_texts, tokenizer, args)

	return train_loader, test_loader

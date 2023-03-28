import logging
logging.basicConfig(level=logging.INFO)


import os
import json
import torch
import random
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from model import GenStyle
from data import create_dataloader_folds
from transformers import GPT2Tokenizer
from sklearn.model_selection import StratifiedKFold
from transformers import get_linear_schedule_with_warmup


def set_seed(args):
	random.seed(args.seed)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	if args.num_gpus > 0:
		torch.cuda.manual_seed_all(args.seed)
		
		



def train_test(args, lang, folds):
	set_seed(args)
	loss_folds={}
	for n,fold in enumerate(folds):
		train_loader, test_loader = fold
		t_total = len(train_loader) // args.accumulate_grad_batches * args.epoch
		genstylemodel = GenStyle(args)
		model = genstylemodel.model
		opt = genstylemodel.configure_optimizers()
		model.to(args.device)
		scheduler = get_linear_schedule_with_warmup(opt, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)

		model.train()
		loss_test = {}

		for epoch in range(args.epoch):
			print(lang +' model, Fold %d of %d, Epoch: %d of %d' %(n+1, args.folds, epoch+1, args.epoch))
			for i,batch in enumerate(tqdm(train_loader)):
				loss = genstylemodel.training_step(batch,i)
				loss /= args.accumulate_grad_batches
				loss.backward()
				if (i+1)%args.accumulate_grad_batches==0:
					torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
					opt.step()
					scheduler.step()
					opt.zero_grad()

			if epoch in range(args.epoch_eval,args.epoch):
				print('Evaluating ...')
				model.eval()
				loss_ep = {}
				for i,batch in enumerate(tqdm(test_loader)):
					with torch.no_grad():
						loss_ep.update(genstylemodel.test_step(batch,i))
				loss_test[epoch+1]=loss_ep
		loss_folds[n+1]=loss_test
				
	return loss_folds

				
				
def main(args):
	tokenizer = GPT2Tokenizer.from_pretrained(args.model_dir)
	tokenizer.pad_token = '<pad>'
	data = pd.read_csv(args.data_path)
	texts, labels = data[['essay']], data[['label']]
	skf = StratifiedKFold(n_splits=args.folds)
	
	
	langs = ['ARA', 'CHI', 'FRE', 'GER', 'HIN', 'ITA', 'JPN', 'KOR', 'SPA', 'TEL', 'TUR' ]
	errors = {}
	for lang in langs[6:]:
		splits= skf.split(texts, labels)
		folds, test_labels = create_dataloader_folds(args, tokenizer, lang, splits, texts, labels)
		errors[lang] = train_test(args, lang, folds)
	
	truth_folds = {i:test_labels[i].tolist() for i in range(args.folds)}
	
	json.dump(errors, open('errors.json','w'))
	json.dump(truth_folds, open('truth.json','w'))
	

				
		



if __name__ == "__main__": 
	parser = GenStyle.add_model_specific_args()
	parser.add_argument('--seed', type=int, default=42)
	parser.add_argument('--num_gpus', type=int, default=2)
	parser.add_argument('--train_batch_size', default= 1, type=int)
	parser.add_argument('--data_path', type=str, default= '../TOEFL-train.csv')
	parser.add_argument('--epoch', type=int, default=3)
	parser.add_argument('--epoch_eval', type=int, default=0)
	parser.add_argument('--max_grad_norm', type=int, default=1)
	parser.add_argument('--warmup_steps', type=int, default=0)

	parser.add_argument('--folds', type=int, default=10)
	parser.add_argument('--device', default= torch.device('cuda:1'))
	parser.add_argument('--accumulate_grad_batches', type=int, default=4)
	parser.add_argument('--resume_from_checkpoint', type=str, default=None)
	args = parser.parse_args()
	main(args)


	
	
	
import gc
import os
import json
import torch
import logging
import numpy as np
import math
#import pytorch_lightning as pl
from argparse import ArgumentParser
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, ConcatDataset
from transformers import get_linear_schedule_with_warmup
from transformers import GPT2Config, GPT2LMHeadModel, AdamW


logger = logging.getLogger(__name__)


class GenStyle(torch.nn.Module):
	def __init__(self, hparams):
		super(GenStyle, self).__init__()
		self.model = GPT2LMHeadModel.from_pretrained(hparams.model_dir)
		self.hparams = hparams
		
	
		
	def forward(self, input_ids, labels):
		out, *_ = self.model(input_ids, labels =labels)
		return out
	

	def training_step(self, batch, batch_idx):
		device = self.hparams.device
		input_ids, labels = batch
		input_ids, labels = input_ids.to(device), labels.to(device)
		return self.forward(input_ids, labels)


	
	def test_step(self, batch, batch_idx):
		device = self.hparams.device
		input_ids, labels, lns = batch
		input_ids, labels = input_ids.to(device), labels.to(device)
		loss = self.forward(input_ids, labels).cpu().item()  #/lns[0].item()
		return loss
		#return {batch_idx : math.exp(loss)}
	


	def configure_optimizers(self):
		no_decay = ["bias", "LayerNorm.weight"]
		optimizer_grouped_parameters = [
			{
				"params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
				"weight_decay": self.hparams.weight_decay,
			},
			{
				"params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 
				"weight_decay": 0.0
			},
		]
		optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate)
		return optimizer
	
	
	

		


	
	
	
	@staticmethod
	def add_model_specific_args():
		home = os.environ['HOME']
		parser = ArgumentParser()
		parser.add_argument('--model_dir', type=str, default='gpt2-medium')
		parser.add_argument('--cache_dir', type=str, default="")
		parser.add_argument('--learning_rate', default=5.25e-5, type=float)
		parser.add_argument('--weight_decay', default=0.0, type=float)
		parser.add_argument('--block_size', type=int, default=512)
		return parser
	
	
	


	
	
	



import os
import csv
import json
import numpy as np
import math
import pandas as pd
from sklearn.metrics import f1_score



def evaluate(errors_path, truth_path):
	errors = json.load(open(errors_path,'r'))
	truth = json.load(open(truth_path,'r'))
	
	langs = list(errors.keys())
	folds = list(errors[langs[0]].keys())
	epochs = list(errors[langs[0]][folds[0]].keys())
	errors_fold_epoch={fold:{epoch:{lang:[] for lang in langs} for epoch in epochs} for fold in folds}

	for lang in errors:
		for fold in errors[lang]:
			for epoch in errors[lang][fold]:
				errors_fold_epoch[fold][epoch][lang]=errors[lang][fold][epoch]


	results = {fold:{} for fold in folds}
	confusion = {fold:{} for fold in folds}

	for fold in folds:
		tr = truth[str(int(fold)-1)]
		for epoch in epochs:
			df = pd.DataFrame(errors_fold_epoch[fold][epoch])
			label = df[langs].apply(np.argmin, axis=1)
			acc = (tr==label).sum()/len(tr)
			f1 = f1_score(tr,label, average='macro')
			results[fold][epoch]=[acc,f1]
			confusion[fold][epoch] = {'pr': label.values.tolist(), 'tr':tr}


	average = {epoch:{'acc':[np.mean([results[i][epoch][0] for i in folds]), np.std([results[i][epoch][0] for i in folds])], 
					  'f1':[np.mean([results[i][epoch][1] for i in folds]) , np.std([results[i][epoch][0] for i in folds])]} for epoch in epochs}
	results.update({'average':average})
	
	json.dump(results, open('result.json','w'))
	json.dump(confusion, open('confusion.json','w'))




if __name__ == '__main__':
	evaluate('errors.json', 'truth.json')
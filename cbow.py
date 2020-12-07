import json
import string
import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.special import expit as sigmoid


def extract_vocab_info(filename,parameters):

	lines = open(filename)

	cnt = 0
	whole_data = list()
	for line in lines:
		cnt += 1
		data = json.loads(line)['reviewText'].split('.')
		for sentence in data:
			if(sentence.strip()):
				sentence = sentence.translate(str.maketrans('', '', string.punctuation)).lower().split()
				sentence_new = list()
				for word in sentence:
					if not any(c.isdigit() for c in word):
						sentence_new.append(word)
				whole_data.append(sentence_new)
		if(cnt == 50000):
			break

	word2cnt = dict()

	for data in whole_data:
		for word in data:
			if word in word2cnt:
				word2cnt[word] += 1
			else:	
				word2cnt[word] = 1

	
	updated_word2cnt = dict()
	min_freq = parameters['min_freq']

	for key in word2cnt:
		if(word2cnt[key] > min_freq):
			updated_word2cnt[key] = word2cnt[key]

	# print(word2cnt)		
	# print(updated_word2cnt)
			
	word2cnt = updated_word2cnt				

	word2index = dict()
	index2word = dict()
	keys = list(word2cnt.keys())

	for index in range(len(keys)):
		word2index[keys[index]] = index
		index2word[index] = keys[index]

	wordfreq = np.zeros(len(word2cnt))

	i = 0
	for word in word2cnt:
		wordfreq[i] = word2cnt[word]
		i+=1

	# print(wordfreq)	
	wordprob = wordfreq**0.75
	# print(wordprob)	

	wordprob = wordprob / wordprob.sum()	
	# print(wordprob)	



	return word2cnt, word2index, index2word, whole_data, wordprob


# def sigmoid(arr):
# 	return 1 / 1 + np.exp(-arr)


# def softmax(arr):
#     return np.exp(arr - np.max(arr)) / np.sum(np.exp(arr - np.max(arr)),axis=0)	


def define_parameters(window_size, initial_learning_rate, final_learning_rate, epochs, embedding_len, num_negsamples, min_freq):
	p = dict()
	p["window_size"] = window_size
	p["initial_learning_rate"] = initial_learning_rate
	p["final_learning_rate"] = final_learning_rate
	p["epochs"] = epochs 
	p["embedding_len"] = embedding_len 
	p["num_negsamples"] = num_negsamples
	p["min_freq"] = min_freq

	return p


def get_context_words(index, data, window_size, word2index, prob_subsampling):
	ws = window_size
	context_words = []

	c_indx = index - 1 
	while(ws != 0):
		if(c_indx >= 0):
			# if(data[c_indx] in word2index and np.random.random() < (1 - prob_subsampling[word2index[data[c_indx]]])):
			if(data[c_indx] in word2index):
				if(word2index[data[c_indx]] not in context_words):
					context_words.append(word2index[data[c_indx]])
					ws -= 1
		else:
			break

		c_indx -= 1	

	
	ws = window_size
	c_indx = index + 1
	while(ws != 0):
		if(c_indx < len(data)):
			if(data[c_indx] in word2index):
			# if(data[c_indx] in word2index and np.random.random() < (1 - prob_subsampling[word2index[data[c_indx]]])):
				if(word2index[data[c_indx]] not in context_words):
					context_words.append(word2index[data[c_indx]])
					ws -= 1
		else:
			break

		c_indx += 1	


	return np.array(context_words)


def forward_backward_propagation(main_word, context_words, negsamples, W_input, W_output, learning_rate, vocab_size):
	# print(W_input)	
	# print('********************8')
	targets = list()
	for context_word in context_words:
		targets.append(context_word)

	for negsample in negsamples:
		targets.append(negsample)

	targets = np.array(targets)

	#forward_propagation

	h = W_input[targets,:]

	prediction = h.dot(W_output[main_word].T)
	prediction = sigmoid(prediction)


	#backward_propagation
	tj = np.zeros(len(targets))
	for i in range(len(context_words)):
		tj[i] = 1

	pred_error = prediction - tj

	# print(h.shape)
	# print(prediction.shape)
	# print(pred_error.shape)
	# print(W_output[:,targets].T.shape)
	# print('*****************************')	

	del_W_input = np.outer(pred_error, W_output[main_word])
	# print(del_W_input.shape)

	# print(h.shape)

	del_W_output = np.dot(pred_error.T,h)

	# print(del_W_output.shape)

	# print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')

	for index in range(len(targets)):
		W_input[targets[index]] -= learning_rate*del_W_input[index]


	W_output[main_word] -= learning_rate*del_W_output

	costr = tj * np.log(prediction + 1e-10) + (1 - tj) * np.log(1 - prediction + 1e-10)

	# cost = pred_error.sum()
	cost = -1*costr.sum()
	# print(W_input.shape())	

	return W_input, W_output, cost

			


def train_model(parameters):
	word2cnt, word2index, index2word, whole_data, wordprob = extract_vocab_info('./reviews_Electronics_5.json' , parameters)

	window_size = parameters["window_size"] 
	initial_learning_rate = parameters["initial_learning_rate"] 
	final_learning_rate = parameters["final_learning_rate"]
	epochs = parameters["epochs"]
	embedding_len = parameters["embedding_len"]  
	num_negsamples = parameters["num_negsamples"]
	min_freq = parameters["min_freq"]
	vocab_size = len(word2index)

	del_lr = (initial_learning_rate - final_learning_rate) / epochs
	learning_rate = initial_learning_rate

	W_input = np.random.randn(vocab_size, embedding_len)
	W_output = np.random.randn(vocab_size, embedding_len)

	# print(W_input)
	# print('@@@@@@@@@@@@@@@@@@@@@@@@')

	costs = []

	prob_subsampling = 1 - np.sqrt(1e-5/wordprob) 

	random.shuffle(whole_data)

	# print(word2index)

	tot_sen = len(whole_data)
	for epoch in range(0,epochs):
		print("Epoch no.",epoch,"started")
		cost = 0
		cnt = 0
		cntfb = 0
		for data in whole_data:
			random_order = np.random.permutation(np.arange(len(data)))
			# print(data)
			# print(random_order)

			for index in random_order:
				# if data[index] in word2index and np.random.random() < (1 - prob_subsampling[word2index[data[index]]]):
				if data[index] in word2index:
					main_word = word2index[data[index]]
					context_words = get_context_words(index, data, window_size, word2index, prob_subsampling)
					if(len(context_words) == 0):
						continue;

					negsamples = np.random.choice(vocab_size, size=num_negsamples , p = wordprob)

					W_input, W_output , ccost = forward_backward_propagation(main_word, context_words, negsamples, W_input, W_output, learning_rate, vocab_size)
					cost += ccost
					cntfb += 1

			cnt += 1

			if(cnt%500 == 0):
				print("sentences remaining: ",tot_sen - cnt)

		costs.append(cost/cntfb)
		print(cost/cntfb)

		embedding = {}

		embedding['word2index'] = word2index
		embedding['index2word'] = index2word
		embedding['cost'] = costs
		embedding['W_output'] = W_output.tolist()

		x = "cbow_epoch" + str(epoch) + ".json"
		out_file = open(x,'w')
		json.dump(embedding, out_file)
		out_file.close()

		learning_rate *= 0.66

	plt.plot(costs)
	plt.show()					




parameters = define_parameters(3, 0.03, 0.0001, 2, 100, 10, 3)
train_model(parameters)






import re
from nltk import word_tokenize
import unicodedata
import string
import numpy as np
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import pandas as pd

class PreprocessData:
	def url_elimination(self, text):
		urls = re.findall('(?:(?:https?|ftp):\/\/)?[\w/\-?=%.]+\.[\w/\-?=%.]+', text)
		output = ''
		for url in urls:
			x = text.find(url)
			if x > 0:
				output += text[:x]
				output += "url "
				text = text[x+len(url) +1:]
		output += text
		return output

	def tokenize(self, text):
		text = self.url_elimination(text)
		return [w.lower() for w in word_tokenize(text)]
		
	def remove_non_ascii(self, words):
		"""Remove non-ASCII characters from list of tokenized words"""
		new_words = []
		for word in words:
			new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
			new_words.append(new_word)
		return new_words

	def remove_punctuation(self, words):
		new_words = []
		for word in words:
			temp = word.strip(string.punctuation)
			if temp is not '':
				new_words.append(temp)
		return new_words

	def replace_numbers(self, words):
		"""Replace all interger occurrences in list of tokenized words with textual representation"""
		return [re.sub(r'\d+', '<num>', word) for word in words]

	def normalize_string(self, text):
		return re.sub(r'([a-z])\1+', lambda m: m.group(1).lower(), text, flags=re.IGNORECASE)

	def clean_str(self, string):
		string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
		string = re.sub(r"\'s", " \'s", string)
		string = re.sub(r"\'ve", " \'ve", string)
		string = re.sub(r"n\'t", " n\'t", string)
		string = re.sub(r"\'re", " \'re", string)
		string = re.sub(r"\'d", " \'d", string)
		string = re.sub(r"\'ll", " \'ll", string)
		string = re.sub(r",", " , ", string)
		string = re.sub(r"!", " ! ", string)
		string = re.sub(r"\(", " \( ", string)
		string = re.sub(r"\)", " \) ", string)
		string = re.sub(r"\?", " \? ", string)
		string = re.sub(r"\s{2,}", " ", string)
		return string.strip().lower()

	def clean(self, text):
		text = self.clean_str(text)
		text = self.normalize_string(text)
		words = self.tokenize(text)
		words = self.remove_non_ascii(words)
		words = self.remove_punctuation(words)
		words = self.replace_numbers(words)
		return ' '.join(words)

	def get_modified_data(self, filepath):
		f = open(filepath, 'r')
		data_processed = []
		for line in f.readlines():
			line = line.strip()
			temp = line.split('\t')
			for i in range(2):
				temp[i] = self.clean(temp[i])
			data_processed.append(temp)
		f.close()
		return data_processed
    
	def build_corpus(self, filepath):
		print('Loading %s' % filepath)
		data_processed = self.get_modified_data(filepath)
		questions = []
		answers = []
		labels = []
		for i in range (len(data_processed)):
			questions.extend([data_processed[i][0]])
			answers.extend([data_processed[i][1]])
			labels.append(int(data_processed[i][2]))
		print('Done loading %s' % filepath)
		return (questions, answers, labels)
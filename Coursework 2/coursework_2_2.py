import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn import metrics

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

import warnings
from sklearn.metrics import classification_report

from datetime import datetime

import json
import sys
from pathlib import Path

from nltk.stem import PorterStemmer

# Used for removing URLs
import re

import nltk
from nltk.corpus import stopwords

TESTING_PERCENTAGE = 20

# This is the based-on-literature part of the coursework. Many papers were investigated for the clarity of their features and models.
# Finally, it was decided that the project will be based on "Detecting Hate Speech and Offensive Language on Twitter using Machine Learning: An N-gram and TFIDF based Approach"
# (https://arxiv.org/pdf/1809.08651.pdf)
# The paper employs and compares many models.
# As the project is meant to be a single algorithm, the "Logistic Regression" algorithm has been chosen.
# This file contains an implementation of that model, along with all the pre-processing and methodology described in the paper.

# A few functions have been re-used from the first project. Over-function decription comments were copied over from the first project (coursework_2_1),
# but the in-function text wasn't copied so as not to over-clutter this file. It is recommended to read the first project first, as it contains greater detail
# of explanations for functions that're used in both.
# Functions which are elaborated upon in the first script are tagged with "*Exists in project 1"

# *Exists in project 1
# The two main lists, words and combined, are pre-setup
# This helps me visualise how the data is laid out as I work on the project between days
# words is the list of all raw texts from tweets
words = {
	"neutral": {
		"text": []
	},
	"sexism": {
		"text": []
	},
	"racism": {
		"text": []
	}
}

# *Exists in project 1
# combined is the list of texts combined with their labels
# This is performed in combine_percentage
combined = {
	"sexism": {
		"total": {
			"labels": [],
			"data":[]
		},
		"training": {
			"labels": [],
			"data": []
		},
		"test": {
			"labels": [],
			"data": []
		}
	},
	"racism": {
		"total": {
			"labels": [],
			"data":[]
		},
		"training": {
			"labels": [],
			"data": [],
			"vectors": []
		},
		"test": {
			"labels": [],
			"data": [],
			"vectors": []
		}
	},
	"all": {
		"total": {
			"labels": [],
			"data":[]
		},
		"training": {
			"labels": [],
			"data": [],
			"vectors": []
		},
		"test": {
			"labels": [],
			"data": [],
			"vectors": []
		}
	}
}

# *Exists in project 1
# This function loads a line-by-line text file
# It's used to load a custom stopwords file
# The file was written manually to remove certain terms - as we know we're dealing with regular offence vs sexism/racism,
# certain words were removed as they can have false implications in this context
def load_list(filename):
	parent_dir = str(Path(__file__).parent)

	f = open(parent_dir + '\\' + filename)
	data = f.read().splitlines()
	return data

# *Exists in project 1
# This function loads the json text files provided
# regualar json loading wasn't used, as the files aren't formatted as proper json
# (instead of { "x" : [a, b, x] } or [a, b, c] the file follows the format of {a, b, x})
def load_data(filename):
	parent_dir = str(Path(__file__).parent)

	data = []
	with open(parent_dir + '\\' + filename + '.json') as f:
		for line in f:
			js = json.loads(line)
			data.append(js['text'])
	return data

# *Exists in project 1
# This function calls the json loader to load all 3 datasets
# Simple enough
def load(data):
	data["neutral"]["text"] = load_data("neutral")
	data["racism"]["text"] = load_data("racism")
	data["sexism"]["text"] = load_data("sexism")

# *Exists in project 1
# This function takes the 3 datasets and combines them into 1, adding a second list with corresponding appropriate labels
# The datasets are added by percentage - for "primaryPercentage" of 50, the resulting list will have 50% of the primary dataset
# the "string" variables provide the labels used
# (in this instance, primaryString is "neutral")
def combine_data_2_percentage(primary, primaryString, primaryPercentage, secondary, secondaryString, secondaryPercentage):
	total = {
		"labels": [],
		"data": []
	}

	primaryCount = int(len(primary) * (primaryPercentage / 100.0))
	secondaryCount = int(len(secondary) * (secondaryPercentage / 100.0))

	for i in range(primaryCount):
		total["data"].append(primary[i])
		total["labels"].append(primaryString)

	for i in range(secondaryCount):
		total["data"].append(secondary[i])
		total["labels"].append(secondaryString)

	return total

# *Exists in project 1
# Same as above but combines 3 lists instead of 2
def combine_data_3_percentage(primary, primaryString, primaryPercentage, secondary, secondaryString, secondaryPercentage, tertiary, tertiaryString, tertiaryPercentage):
	total = {
		"labels": [],
		"data": []
	}

	primaryCount = int(len(primary) * (primaryPercentage / 100.0))
	secondaryCount = int(len(secondary) * (secondaryPercentage / 100.0))
	tertiaryCount = int(len(tertiary) * (tertiaryPercentage / 100.0))

	for i in range(primaryCount):
		total["data"].append(primary[i])
		total["labels"].append(primaryString)

	for i in range(secondaryCount):
		total["data"].append(secondary[i])
		total["labels"].append(secondaryString)

	for i in range(tertiaryCount):
		total["data"].append(tertiary[i])
		total["labels"].append(tertiaryString)

	return total

# *Exists in project 1
# This function reshuffles all the raw datasets and then combines them all into appropriate lists
# Reshuffling:
#	As only a certain percentage of each dataset is to be used, the dataset must be shuffled to ensure that, over multiple iterations, 50% of data won't always return the same 50%
def combine_percentage(data, combined, primaryPercentage, secondaryPercentage, tertiaryPercentage):
	random.shuffle(data["neutral"]["text"])
	random.shuffle(data["sexism"]["text"])
	random.shuffle(data["racism"]["text"])

	combined["sexism"]["total"] = combine_data_2_percentage(data["neutral"]["text"], "neutral", primaryPercentage, data["sexism"]["text"], "sexism", secondaryPercentage)
	combined["racism"]["total"] = combine_data_2_percentage(data["neutral"]["text"], "neutral", primaryPercentage, data["racism"]["text"], "racism", secondaryPercentage)
	combined["all"]["total"] = combine_data_3_percentage(data["neutral"]["text"], "neutral", primaryPercentage, data["sexism"]["text"], "sexism", secondaryPercentage, data["racism"]["text"], "racism", tertiaryPercentage)

# *Exists in project 1
# The combined dataset contains separate lists of data and labels. Shuffling this isn't as easy as doing random.shuffle() on both because that will result in labels no longer aligning
# The data could be paired before shuffling, but it seems this approach works slightly faster.
# Shuffling is performed by generating a list of indices, shuffling that list, and then organising the label and data lists with those indices.
def shuffle(data):
	output = {
		"labels": [],
		"data": []
	}

	count = len(data["data"])
	indices = list(range(count))

	random.shuffle(indices)

	for i in indices:
		output["labels"].append(data["labels"][i])
		output["data"].append(data["data"][i])

	return output

# *Exists in project 1
# Taken from the practicals
# Takes a list of result values and returns the average
def get_average_results(report_list):
	totals = []
	averages = []

	#for item in report_list:
	for i in range(len(report_list)):
		if (i == 0):
			totals = [0] * len(report_list[0])
			averages = [0] * len(report_list[0])

		for j in range(len(report_list[i])):
			totals[j] += report_list[i][j]

	output = 0
	
	if (len(totals) > 0):
		for i in range(len(totals)):
			averages[i] += totals[i] / len(report_list)

		for i in range(len(averages)):
			output += averages[i]
	
		output /= len(totals)
	
	return output

# *Exists in project 1
# Taken from the practicals
# Takes a list of accuracies and returns the average
def get_average_accuracy(report_list):
	total = 0

	for item in report_list:
		total += item

	score = total/len(report_list)
	
	return score

# Takes a dataset and replaces it with a lowercase version of itself
def lowercase(data):
	for i in range(len(data)):
		data[i] = data[i].lower()

# Takes a dataset and removes any URLs, mentions and re-tweet symbols present
def remove_urls(data):
	for i in range(len(data)):
		data[i] = re.sub(r'(^(RT @[a-zA-Z0-9_]+: ))|(@[a-zA-Z0-9_]+)|(\w+:\/\/\S+)', '', data[i], flags=re.MULTILINE)
		data[i] = data[i].lstrip() #remove Space Pattern

# Takes a dataset and removes any stopwords present
# stop_words is a variable in case there is ever a need to test different stopword corpuses
def remove_stopwords(data, stop_words):
	for i in range(len(data)):
		data[i] = " ".join([word for word in data[i].split() if word not in stop_words])

# Uses the Porter Stemmer algorithm to stem all words within a dataset.
# This "normalises" different versions of a word
# For example, "Normally," Normal" and "Norm" would all become "Norm."
# This makes context prediction easier - different versions of words like that mean the same thing,
# and this functions lessens the impact of these spelling differences from the algorithm.
def stem(data):
	ps = PorterStemmer()

	# Iterate over all texts in the dataset.
	# For each text, split it up into words, stem each word, then join them back together.
	for i in range(len(data)):
		data[i] = " ".join([ps.stem(word) for word in data[i].split()])

# This function encompasses everything the writers defined in their "pre-processing" phase.
# The writers used this to sanitse data by removing any small differences in words, or altogether meaningless words, which don't affect context
# This is done by:
#	lowercasing all text
#	Removing all URLs, mentions (@xyz) and retweet symbols (RE @xyz:) from the text
#	Removing all stopwords from the text
#	Stemming (normalizing) all words in every text
#		This different versions of the same word are treated as the same word within the algorithm. i.e.,
#		"Believeing," "Believes" and "Believe" will all count as the same token. This makes it easier to determine intent and context
def pre_process(data, stop_words):
	lowercase(data["neutral"]["text"])
	lowercase(data["racism"]["text"])
	lowercase(data["sexism"]["text"])

	remove_urls(data["neutral"]["text"])
	remove_urls(data["racism"]["text"])
	remove_urls(data["sexism"]["text"])

	# The paper specifies removal of stop-words as a pre-processing step
	# As the writers were also using sklearn, they would have known of sklearn's use of the "stopwords=<stop_words>" parameter and have elected not to use it
	# Otherwise, like with grid searching and cross-validation, they would have stated the use of stopwords in the Model section rather than the pre-processing section
	# Thus, stop-words are removed here, too
	remove_stopwords(data["neutral"]["text"], stop_words)
	remove_stopwords(data["racism"]["text"], stop_words)
	remove_stopwords(data["sexism"]["text"], stop_words)

	stem(data["neutral"]["text"])
	stem(data["racism"]["text"])
	stem(data["sexism"]["text"])

# *Exists in project 1
# Takes a dataset and returns the x'th slice of it when divided into x_count portions
def create_test_train_section(data, x, x_count):
	slice_train_size = int(len(data["training"]["data"]) / (x_count))
	slice_test_size = int(len(data["test"]["data"]) / (x_count))

	test_data = []
	test_labels = []
	train_data = []
	train_labels = []

	if x == x_count - 1:
		test_data = data["test"]["data"][(slice_test_size * x):]
		test_labels = data["test"]["labels"][(slice_test_size * x):]

		train_data = data["training"]["data"][(slice_train_size * x):]
		train_labels = data["training"]["labels"][(slice_train_size * x):]
	else:
		test_data = data["test"]["data"][(slice_test_size * x):(slice_test_size * (x + 1))]
		test_labels = data["test"]["labels"][(slice_test_size * x):(slice_test_size * (x + 1))]

		train_data = data["training"]["data"][(slice_train_size * x):(slice_train_size * (x + 1))]
		train_labels = data["training"]["labels"][(slice_train_size * x):(slice_train_size * (x + 1))]
    
	return train_data, test_data, train_labels, test_labels

# The writers decribed using a grid search function to optimise the parameters of their Logical Regression classifier.
# While this can't be done live (test data is required to rate the parameters), a function was provided to find these parameters.
# This function was run with a "repetitions" of 10 to determine the parameters that are used right now. 10 repetitions were used
# because the optimal parameters depend on the dataset - the most common "best" parameters were determined over 10 iterations.
# Warning: This compares all the possible parameters together by running each possible combination (24 runs to cross-compare all parameters).
# This took 5-9 minutes on a relatively fast computer per iteration, 10 of which are run.
def generate_best_parameters(data_words, data_combined, tested_dataset, repetitions):
	# "We train each model on training dataset by performing grid search for all the combinations of feature parameters"
	# 		GridSearchCV is used to search all working combinations
	# 		(some are split up due to coding differences, eg. "liblinear" doesn't work with "multinomial")
	# "()...) and perform 10-fold cross-validation"
	# 		cv = 10 for GridSearchCV
	warnings.filterwarnings("ignore")

	print("Generating Best Parameters (" + str(repetitions) + " Attempts):")
	for i in range(repetitions):
		# This process can take a while. To help determine runtime, elapsed time per iteration is calculated.
		# This lets us run the function and wait for one iteration to complete before knowing how long we must wait for the full
		# process to complete (roughly 10x one iteration)
		start_time = datetime.now()
		print("")
		print("[" + str(i + 1) + "/" + str(repetitions) + "] Shuffling and Vectorising Data...")
		combine_percentage(data_words, data_combined, 50, 100, 100)

		# In the first program, I used my own version of this. I preferred the way I wrote the code as the data isn't perfect and so I could make up for it
		# Here, however, the writers explicitely stated they used sklearn - thus, it is assumed they would have used sklearn's own train_test_split function
		x_train, x_test, y_train, y_test = train_test_split(data_combined[tested_dataset]["total"]["data"], data_combined[tested_dataset]["total"]["labels"], test_size=(TESTING_PERCENTAGE / 100.0))

		data_combined[tested_dataset]["training"]["data"] = x_train
		data_combined[tested_dataset]["training"]["labels"] = y_train

		data_combined[tested_dataset]["test"]["data"] = x_test
		data_combined[tested_dataset]["test"]["labels"] = y_test

		# The TfidVectoriser combines the CountVectoriser and TfidTransformer
		# "We consider unigram, bigram and trigram features" :: ngram_range = (1, 3)
		# Whether character or word n-grams are used isn't defined, however "the TFIDF approach on the bagof-words features also show promising results" suggests word n-ngrams :: analyzer = "word"
		vectorizer = TfidfVectorizer(use_idf=True, ngram_range=(1,3), analyzer="word")
		data_combined[tested_dataset]["training"]["vectors"] = vectorizer.fit_transform(data_combined[tested_dataset]["training"]["data"])
		data_combined[tested_dataset]["test"]["vectors"] = vectorizer.transform(data_combined[tested_dataset]["test"]["data"])
		
		print("[" + str(i + 1) + "/" + str(repetitions) + "] Performing Grid Search...")
		param_grid = [
  			{'penalty': ['l1'], 'solver': ['liblinear'], 'multi_class': ['ovr'], 'class_weight': [None, 'balanced']},
  			{'penalty': ['l1'], 'solver': ['saga'], 'multi_class': ['ovr', 'multinomial'], 'class_weight': [None, 'balanced']},
  			{'penalty': ['l2'], 'solver': ['liblinear'], 'multi_class': ['ovr'], 'class_weight': [None, 'balanced']},
  			{'penalty': ['l2'], 'solver': ['newton-cg', 'lbfgs', 'sag', 'saga'], 'multi_class': ['ovr', 'multinomial'], 'class_weight': [None, 'balanced']}
 		]

		# To run a grid search, sklearn needs to know what classifier we're using.
		# It's set up to run with a Logistic Regression classifier, with a cross-validation value of 10 (taken from the paper)
		# "The hyperparameters of two algorithms giving best results are tuned for their respective feature parameters, which gives the best result"
		#		The parameters used for LogisticRegression are the results of reg_cv.best_params_
		reg = LogisticRegression()
		reg_cv = GridSearchCV(reg, param_grid, cv=10)
		# Using the .fit function of the grid searcher instead of the classifier makes it perform cross-compare runs with all the parameters
		reg_cv.fit(data_combined[tested_dataset]["training"]["vectors"], data_combined[tested_dataset]["training"]["labels"])
		end_time = datetime.now()

		elapsed_time = end_time - start_time

		print("[" + str(i + 1) + "/" + str(repetitions) + "] Best Parameters :: ", reg_cv.best_params_)
		print("[" + str(i + 1) + "/" + str(repetitions) + "] Accuracy :: ", reg_cv.best_score_)
		print("[" + str(i + 1) + "/" + str(repetitions) + "] Elapsed Time :: ", str(elapsed_time))

# *Exists in project 1
# Based on lectures and practical Jupyter Notebooks
# This function performs a k-fold evaluation of the model by splitting the data into k-portions, training/testing appropriately with each,
# and determining the average accuracy.
# While in most circumstances this would be a good approach, the size of the dataset provided is extremely limiting
# Splitting the dataset into 10 sets results in roughly 1150 "neither" data, 197 "racism" data and 343 "sexism" data.
# This isn't quite enough to successfully train the algorithm.
# It was implemented for comparison purposes, but for the more successful algorithm see "train_predict_count"
def evaluate_model(data_words, data_combined, tested_dataset, folds):
	precision = []
	recall = []
	f1 = []
	accuracy = []

	reports = []

	print("Generating " + str(folds) + "-Fold Evaluation...")
	warnings.filterwarnings("ignore")

	combine_percentage(data_words, data_combined, 50, 100, 100)
	# In the first program, I used my own version of this. I preferred the way I wrote the code as the data isn't perfect and so I could make up for it
	# Here, however, the writers explicitely stated they used sklearn - thus, it is assumed they would have used sklearn's own train_test_split function
	x_train, x_test, y_train, y_test = train_test_split(data_combined[tested_dataset]["total"]["data"], data_combined[tested_dataset]["total"]["labels"], test_size=(TESTING_PERCENTAGE / 100.0))

	data_combined[tested_dataset]["training"]["data"] = x_train
	data_combined[tested_dataset]["training"]["labels"] = y_train

	data_combined[tested_dataset]["test"]["data"] = x_test
	data_combined[tested_dataset]["test"]["labels"] = y_test

	for i in range(folds):
		print("[" + str(i + 1) + " / " + str(folds) + "]")
		x_train, x_test, y_train, y_test = create_test_train_section(data_combined[tested_dataset], i, folds)

		# The TfidVectoriser combines the CountVectoriser and TfidTransformer
		# "We consider unigram, bigram and trigram features" :: ngram_range = (1, 3)
		# Whether character or word n-grams are used isn't defined, however "the TFIDF approach on the bagof-words features also show promising results" suggests word n-ngrams :: analyzer = "word"
		vectorizer = TfidfVectorizer(use_idf=True, ngram_range=(1,3), analyzer="word")
		data_combined[tested_dataset]["training"]["vectors"] = vectorizer.fit_transform(x_train)
		data_combined[tested_dataset]["test"]["vectors"] = vectorizer.transform(x_test)
		
		# To run a grid search, sklearn needs to know what classifier we're using.
		# It's set up to run with a Logistic Regression classifier, with a cross-validation value of 10 (taken from the paper)
		regression = LogisticRegression(class_weight="balanced", multi_class="multinomial", penalty="l2", solver="saga")
		regression.fit(data_combined[tested_dataset]["training"]["vectors"], y_train)
		prediction = regression.predict(data_combined[tested_dataset]["test"]["vectors"])

		report_class = classification_report(y_test, prediction)
		print(report_class)

		report = metrics.precision_recall_fscore_support(y_test, prediction)
		report_accuracy = accuracy_score(y_test, prediction)

		reports.append(report)
		accuracy.append(report_accuracy)

	for i in range(len(reports)):
		precision.append(reports[i][0])
		recall.append(reports[i][1])
		f1.append(reports[i][2])

	precision_result = get_average_results(precision)
	recall_result = get_average_results(recall)
	f_score_result = get_average_results(f1)
	accuracy_result = get_average_accuracy(accuracy)

	print("Precision: ", precision_result)
	print("Recall: ", recall_result)
	print("F-score: ", f_score_result)
	print("Accuracy: ", accuracy_result)

# *Exists in project 1 in a different form. The basic form us described there, the changes are described here.
# This is the secondary evaluation function.
# The dataset provided is small - splittin the dataset into 10 portions leaves too little data to properly train the classifier (the classifier is trained, but not enough to have a high success rate)
# Instead this function takes a slightly different approach.

# Every iteration (every "fold"), the raw data is re-shuffled and re-combined into data/label arrays (using combine_percentage)
# Only 50% of the neutral data is used - there is more than 3x as much neutral data as there is racist/sexist data, thus neutral n-grams can become too heavily weighted
# Using only 50% of the neutral data reduces the impact of this. Furthermore, as reshuffling is performed before combination, the neutral dataset is very likely different
# between every iteration.
# The data/label array combination is also preceeded by shuffling.
# This lowers the negative impact of re-using the same dataset as it won't be exactly the same (some neutral data will be different), and the data will be shuffled between testing/training datasets
# As mentioned, this also allows the entire dataset to be used in training/testing. Allowing 10x the data makes the results far more accurate (shown in report)
def train_predict_count(data_words, data_combined, tested_dataset, repetitions):
	precision = []
	recall = []
	f1 = []
	accuracy = []

	reports = []

	print("Generating [" + str(repetitions) + "] Predictions...")

	warnings.filterwarnings("ignore")
	# The TfidVectoriser combines the CountVectoriser and TfidTransformer
	# "We consider unigram, bigram and trigram features" :: ngram_range = (1, 3)
	# Whether character or word n-grams are used isn't defined, however "the TFIDF approach on the bagof-words features also show promising results" suggests word n-ngrams :: analyzer = "word"

	# In the first program, I used my own version of this. I preferred the way I wrote the code as the data isn't perfect and so I could make up for it
	# Here, however, the writers explicitely stated they used sklearn - thus, it is assumed they would have used sklearn's own train_test_split function

	for i in range(repetitions):
		print("[" + str(i + 1) + " / " + str(repetitions) + "]")
		# In future projects, this should be changed: right now, the program re-combines all datasets. This takes more time than necessary.
		# A quick edit to only re-combine one specific dataset could save a lot of processing time.
		combine_percentage(data_words, data_combined, 50, 100, 100)
		# In the first program, I used my own version of this. I preferred the way I wrote the code as the data isn't perfect and so I could make up for it
		# Here, however, the writers explicitely stated they used sklearn - thus, it is assumed they would have used sklearn's own train_test_split function
		x_train, x_test, y_train, y_test = train_test_split(data_combined[tested_dataset]["total"]["data"], data_combined[tested_dataset]["total"]["labels"], test_size=(TESTING_PERCENTAGE / 100.0))

		data_combined[tested_dataset]["training"]["data"] = x_train
		data_combined[tested_dataset]["training"]["labels"] = y_train

		data_combined[tested_dataset]["test"]["data"] = x_test
		data_combined[tested_dataset]["test"]["labels"] = y_test

		# The TfidVectoriser combines the CountVectoriser and TfidTransformer
		# "We consider unigram, bigram and trigram features" :: ngram_range = (1, 3)
		# Whether character or word n-grams are used isn't defined, however "the TFIDF approach on the bagof-words features also show promising results" suggests word n-ngrams :: analyzer = "word"
		vectorizer = TfidfVectorizer(use_idf=True, ngram_range=(1,3), analyzer="word")
		data_combined[tested_dataset]["training"]["vectors"] = vectorizer.fit_transform(data_combined[tested_dataset]["training"]["data"])
		data_combined[tested_dataset]["test"]["vectors"] = vectorizer.transform(data_combined[tested_dataset]["test"]["data"])

		# To run a grid search, sklearn needs to know what classifier we're using.
		# It's set up to run with a Logistic Regression classifier, with a cross-validation value of 10 (taken from the paper)
		regression = LogisticRegression(class_weight="balanced", multi_class="multinomial", penalty="l2", solver="saga")
		regression.fit(data_combined[tested_dataset]["training"]["vectors"], data_combined[tested_dataset]["training"]["labels"])
		prediction = regression.predict(data_combined[tested_dataset]["test"]["vectors"])

		report_class = classification_report(data_combined[tested_dataset]["test"]["labels"], prediction)
		print(report_class)

		report = metrics.precision_recall_fscore_support(data_combined[tested_dataset]["test"]["labels"], prediction)
		report_accuracy = accuracy_score(data_combined[tested_dataset]["test"]["labels"], prediction)

		reports.append(report)
		accuracy.append(report_accuracy)

	for i in range(len(reports)):
		precision.append(reports[i][0])
		recall.append(reports[i][1])
		f1.append(reports[i][2])

	precision_result = get_average_results(precision)
	recall_result = get_average_results(recall)
	f_score_result = get_average_results(f1)
	accuracy_result = get_average_accuracy(accuracy)

	print("Precision: ", precision_result)
	print("Recall: ", recall_result)
	print("F-score: ", f_score_result)
	print("Accuracy: ", accuracy_result)

# This project works similar to courswork_2_1, but instead of SVM it uses Logistic Regression. The Logistic Regression parameters are determined by an
# exhaustive 10-fold grid-search algorithm. The dataset itself starts off the same as coursework_2_1, but is sanitised by a number of pre-processing techniques to remove
# formatting text and unnecessary words, as well as normalising terms to increase their relevance and improve context-understanding.
# In short, while the fitting and transforming code itself is similar (though changes to work with Logistic Regression), the power of this algorithm
# instead comes from the amount of data pre-processing and model fine-tuning that comes beforehand.
# To avoid a wall of green text, the reason for and explanation of the functions themselves is done within/above each function
def main():
	print("Generating Data...")
	load(words)

	# The exact stop-words weren't defined in the paper.
	# It's unlikely I could come up with the same corpus the paper used. However, it is known
	# that the paper uses sklearn. Thus, it is very likely that the nltk english stopword corpus was used.
	stop_words = set(stopwords.words('english'))

	# The data is pre-processed in accordance to the writers' instructions
	pre_process(words, stop_words)

	# The dataset used.
	#	"all" = all datasets combined
	#	"racism" = dataset comprising neutral + racist tweets
	#	"sexism" = dataset comprising neutral + sexist tweets
	tested_dataset = "all"
	repetitions = 2

	# Use any of these functions
	# Generate Best Parameters : use grid search to find the best model parameters for this data set
	# Evaluate Model : Run a 10-fold evaluation of the model. The data will be split into 10 portions
	# Generate Predictions: : Run a single run of the model with pre-set parameters (taken from the best outcomes of 10 "generate_best_parameters" runs.)

	#generate_best_parameters(words, combined, tested_dataset, repetitions)
	#evaluate_model(words, combined, tested_dataset, repetitions)
	train_predict_count(words, combined, tested_dataset, repetitions)


main()
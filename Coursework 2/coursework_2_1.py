import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn import metrics

import warnings
from sklearn.metrics import classification_report

import json
import sys
from pathlib import Path

# This is the first part of the project, the practical-inspired algorithm.
# To analyse this, I recommend starting with main() (at the very bottom)
# and following the flow of control. Detailed decriptions are provided the first time something is done (or whenever appropriate),
# thus following the flow of control will make understanding the project easier.
# If using Visual Studio Code, ctrl+clicking on called functions is helpful

# An equivalent to a global variable that can be changed at the top of the script
# What percentage of the dataset should be used for testing?
# default=20 (20%)
TESTING_PERCENTAGE = 20
# This value was chosen because the dataset is fairly small, so the more trainable data is used, the better
# The downside to is that the resulting predictions accuracies can vary wildly (smaller data set, less chance to dilute any edge cases)
# This should be visible in the results by more jumpy results across many testing runs

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

# This function loads a line-by-line text file
# It's used to load a custom stopwords file
# The file was written manually to remove certain terms - as we know we're dealing with regular offence vs sexism/racism,
# certain words were removed as they can have false implications in this context
def load_list(filename):
	parent_dir = str(Path(__file__).parent)

	f = open(parent_dir + '\\' + filename)
	data = f.read().splitlines()
	return data

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

# This function calls the json loader to load all 3 datasets
# Simple enough
def load(data):
	data["neutral"]["text"] = load_data("neutral")
	data["racism"]["text"] = load_data("racism")
	data["sexism"]["text"] = load_data("sexism")

# This function takes the 3 datasets and combines them into 1, adding a second list with corresponding appropriate labels
# The datasets are added by percentage - for "primaryPercentage" of 50, the resulting list will have 50% of the primary dataset
# the "string" variables provide the labels used
# (in this instance, primaryString is "neutral")
def combine_data_2_percentage(primary, primaryString, primaryPercentage, secondary, secondaryString, secondaryPercentage):
	total = {
		"labels": [],
		"data": []
	}

	# Calculate how many elements from each list to use (if primaryPercentage = 20, use 20% of primary)
	primaryCount = int(len(primary) * (primaryPercentage / 100.0))
	secondaryCount = int(len(secondary) * (secondaryPercentage / 100.0))
	
	# Add all values from each list consecutively
	# This is done because of the nature of shuffling:
	# python's randomiser is fairly even-ish. If we combine the lists by interlacing them
	#	[neutral, racist, neutral, racist, (...)]
	# instead of making one follow the other
	#	[neutral, neutral, (...), neutral, racist, racist, (...)]
	# then shuffling them later will almost definitely heavily unbalance the weighting of each type of text between the training/testing datasets
	# if, instead, the lists follow each other, the shuffling will result in a move evenly shuffled list.
	for i in range(primaryCount):
		total["data"].append(primary[i])
		total["labels"].append(primaryString)

	for i in range(secondaryCount):
		total["data"].append(secondary[i])
		total["labels"].append(secondaryString)

	return total

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

# ===========================================================================================================
# ======================================== <EXTRA COMBINE FUNCTIONS> ========================================
# ===========================================================================================================
# Not used in the project, but these functions were saved as they can be used interestingly.
# combine_data_3 combines all elements from 3 lists with the appropriate labels
# It's equivalent to combine_data_3_percentage with all percentages set to 100

def combine_data_2(primary, primaryString, secondary, secondaryString):
	total = {
		"labels": [],
		"data": []
	}

	for text in primary:
		total["data"].append(text)
		total["labels"].append(primaryString)
		
	for text in secondary:
		total["data"].append(text)
		total["labels"].append(secondaryString)

	return total

# Same as above but combines 3 lists instead of 2
def combine_data_3(primary, primaryString, secondary, secondaryString, tertiary, tertiaryString):
	total = {
		"labels": [],
		"data": []
	}

	for text in primary:
		total["data"].append(text)
		total["labels"].append(primaryString)
		
	for text in secondary:
		total["data"].append(text)
		total["labels"].append(secondaryString)

	for text in tertiary:
		total["data"].append(text)
		total["labels"].append(tertiaryString)

	return total

# combine_data_3_equal creates a combination list of data/labels whose elements come from all raw lists in equal amounts
# It does this by finding the smallest list size, min_size then extracting min_size elements from all raw lists
# This wasn't used as one of the datasets was far smaller than the others - limiting all elements to min_size would
# result in a tiny combination list
def combine_data_2_equal(primary, primaryString, secondary, secondaryString):
	total = {
		"labels": [],
		"data": []
	}

	# min_size = size of smallest component list
	min_size = min(len(primary), len(secondary))

	for i in range(min_size):
		total["data"].append(primary[i])
		total["labels"].append(primaryString)

	for i in range(min_size):
		total["data"].append(secondary[i])
		total["labels"].append(secondaryString)

	return total

# Same as above but combines 3 lists instead of 2
def combine_data_3_equal(primary, primaryString, secondary, secondaryString, tertiary, tertiaryString):
	total = {
		"labels": [],
		"data": []
	}

	min_size = min(min(len(primary), len(secondary)), len(tertiary))

	for i in range(min_size):
		total["data"].append(primary[i])
		total["labels"].append(primaryString)

	for i in range(min_size):
		total["data"].append(secondary[i])
		total["labels"].append(secondaryString)

	for i in range(min_size):
		total["data"].append(tertiary[i])
		total["labels"].append(tertiaryString)

	return total

# combine is equivalent to combine_percentage with all percentages set to 100%
def combine(data, combined):
	random.shuffle(data["neutral"]["text"])
	random.shuffle(data["sexism"]["text"])
	random.shuffle(data["racism"]["text"])

	combined["sexism"]["total"] = combine_data_2(data["neutral"]["text"], "neutral", data["sexism"]["text"], "sexism")
	combined["racism"]["total"] = combine_data_2(data["neutral"]["text"], "neutral", data["racism"]["text"], "racism")
	combined["all"]["total"] = combine_data_3(data["neutral"]["text"], "neutral", data["sexism"]["text"], "sexism", data["racism"]["text"], "racism")

# combine_equal works like an auto-scaling combine_percentage
# The resultant lists hold equal amounts of each group
# This is done by finding the smallest list (of size Ns) then extracting Ns elements of each list to combine into the new lists
# Such that the resulting lists be of size 3Ns
def combine_equal(data, combined):
	random.shuffle(data["neutral"]["text"])
	random.shuffle(data["sexism"]["text"])
	random.shuffle(data["racism"]["text"])

	combined["sexism"]["total"] = combine_data_2_equal(data["neutral"]["text"], "neutral", data["sexism"]["text"], "sexism")
	combined["racism"]["total"] = combine_data_2_equal(data["neutral"]["text"], "neutral", data["racism"]["text"], "racism")
	combined["all"]["total"] = combine_data_3_equal(data["neutral"]["text"], "neutral", data["sexism"]["text"], "sexism", data["racism"]["text"], "racism")
# ===========================================================================================================
# ======================================= </EXTRA COMBINE FUNCTIONS\> =======================================
# ===========================================================================================================

# The combined dataset contains separate lists of data and labels. Shuffling this isn't as easy as doing random.shuffle() on both because that will result in labels no longer aligning
# The data could be paired before shuffling, but it seems this approach works slightly faster.
# Shuffling is performed by generating a list of indices, shuffling that list, and then organising the label and data lists with those indices.
def shuffle(data):
	output = {
		"labels": [],
		"data": []
	}

	# Generate list of indices of the same size as the dataset ([0 ... data["data"].size - 1])
	# Both data["data"] and data["labels"] have an equal amount of elements - the labels label the data
	count = len(data["data"])
	indices = list(range(count))

	# Shuffle the list of indices
	random.shuffle(indices)

	# Organise the return list in accordance with the shuffled index list
	for i in indices:
		output["labels"].append(data["labels"][i])
		output["data"].append(data["data"][i])

	return output

# Takes a dataset of two lists and returns them as two datasets of two lists each, split by percentage
# The resulting lists are randomly taken
def split_data(data, testing_percentage):
	# Shuffle the dataset
	# Custom function used because this is a dataset of two lists, whose indices must still match
	data = shuffle(data)

	# Determine the index at which to split (by default, 20% through)
	split_point = int((len(data["data"]) / (100 / testing_percentage)))

	output_0 = { "labels": [], "data":[] }
	output_1 = { "labels": [], "data":[] }

	output_0["labels"] = data["labels"][split_point:]
	output_0["data"] = data["data"][split_point:]

	output_1["labels"] = data["labels"][:split_point]
	output_1["data"] = data["data"][:split_point]
	
	return (output_0, output_1)

# Splits all datasets within this list into training/testing lists
def split(data):
	data["sexism"]["training"], data["sexism"]["test"] = split_data(data["sexism"]["total"], TESTING_PERCENTAGE)
	data["racism"]["training"], data["racism"]["test"] = split_data(data["racism"]["total"], TESTING_PERCENTAGE)
	data["all"]["training"], data["all"]["test"] = split_data(data["all"]["total"], TESTING_PERCENTAGE)

# Taken from the practicals
# Takes a list of result values and returns the average
def get_average_results(report_list):
	totals = []
	averages = []

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

# Taken from the practicals
# Takes a list of accuracies and returns the average
def get_average_accuracy(report_list):
	total = 0

	for item in report_list:
		total += item

	score = total/len(report_list)
	
	return score

# Takes a dataset and returns the x'th slice of it when divided into x_count portions
def create_test_train_section(data, x, x_count):
	# The training and testing dataset size will be different.
	# The size of data and label lists within each dataset will be the same
	# Slices are thus generated to match
	slice_train_size = int(len(data["training"]["data"]) / (x_count))
	slice_test_size = int(len(data["test"]["data"]) / (x_count))

	test_data = []
	test_labels = []
	train_data = []
	train_labels = []

	# It's possible that the dataset can't be divided equally (55 / 3, for example)
	# To still return all the data, make sure that the final slice returns the data from (current_slice) to the end of the array
	# Otherwise, return all data from (current_slice) to (current_slice + 1)
	# current_slice (x) is a value from 0 to slice_count (x_count) - 1.
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

# This used to be two functions (the for-loop called a secondary function to perform the each singular classification)
# This, however, made the algorithm harder to explain as the helper function should be above this one. Instead, they were merged into the same function
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
def train_predict_count(data_words, data_combined, tested_dataset, stop_words, count):
	precision = []
	recall = []
	f1 = []
	accuracy = []

	reports = []

	for i in range(count):
		print("[" + str(i + 1) + " / " + str(count) + "]")
		# combine_percentage takes the raw tweet texts from 3 arrays (neutral, racism, sexism) and cross-combines them
		# and adds appropriate labels. It results in 3 new arrays (neutral+racism, neutral+sexism, neutral+racism+sexism)
		# Furthermore, this function shuffles the raw lists before combining them. This allows the same lists to be re-used with minimal impact on prediction
		# split then splits the data into training and testing datasets. The data is shuffled before this is done.
		combine_percentage(data_words, data_combined, 50, 100, 100)
		split(data_combined)

		# This is the standard vectoriser used in the practicals. min_df was reduced to 1 to allow even individual n-grams to have an impact
		# This was done because, in this specific context, we're not detecting abritrary hateful speech - we're detecting racism/sexism
		# Unlike regular offensive language, racism/sexism often employs specific offensive words which, in this case, would be unigrams.
		# As these words can be obscure (and the dataset is fairly small), it's highly likely that rare offensive words can be used.
		# Thus, we should allow even single-occurances to impact the results.
		# This has in general improved prediction ratings.

		# TfidVectorizer is a CountVectorizer combined with a TfidTransformer
		# CountVectorizer converts string data into a matrix of token counts - only numeric data can be worked on by the classifier
		# TfidTransformer transforms the resultant count matrix into its tf-idf representation, i.e. scales the values appropriately

		# Given that sexist language is gender-specific (as well as possibly transphobic), it was decided that stopwords won't be used. They would eliminate potentially meaningful
		# interactions ("She", "They", "Them", etc.)
		# Furthermore, context in sexist and racist abuse stems from historic euphemism - multiple words put together change potentially harmless words ("Get" "Kitchen" "Go" "Back" "To")
		# thus, it is clear that n-grams of a higher order than 1 must be used.
		# Upon experimenting, n-grams of size 1-3 returned good results.
		#ngrams = (1,3)
		ngrams = (1,1)
		warnings.filterwarnings("ignore")

		vectorizer = TfidfVectorizer(min_df=1, sublinear_tf=True, ngram_range=ngrams)
		data_combined[tested_dataset]["training"]["vectors"] = vectorizer.fit_transform(data_combined[tested_dataset]["training"]["data"])
		data_combined[tested_dataset]["test"]["vectors"] = vectorizer.transform(data_combined[tested_dataset]["test"]["data"])

		# The classifier can use stop_words.
		# It was decided that they shouldn't be used as extensive testing showed that, on average, using stop-words in this model reduces prediction accuracy
		#classifier = svm.SVC(kernel='linear', gamma="scale", stop_words=stop_words)
		classifier = svm.SVC(kernel='linear', gamma="scale")
		classifier.fit(data_combined[tested_dataset]["training"]["vectors"], data_combined[tested_dataset]["training"]["labels"])
		prediction = classifier.predict(data_combined[tested_dataset]["test"]["vectors"])

		report_class = classification_report(data_combined[tested_dataset]["test"]["labels"], prediction)
		print(report_class)

		report = metrics.precision_recall_fscore_support(data_combined[tested_dataset]["test"]["labels"], prediction)
		report_accuracy = accuracy_score(data_combined[tested_dataset]["test"]["labels"], prediction)

		reports.append(report)
		accuracy.append(report_accuracy)

	# The data is extracted from the reports so that they may be averaged
	for i in range(len(reports)):
		precision.append(reports[i][0])
		recall.append(reports[i][1])
		f1.append(reports[i][2])

	# The reports have been taken from 10 runs - average these results
	precision_result = get_average_results(precision)
	recall_result = get_average_results(recall)
	f_score_result = get_average_results(f1)
	accuracy_result = get_average_accuracy(accuracy)

	print("Precision: ", precision_result)
	print("Recall: ", recall_result)
	print("F-score: ", f_score_result)
	print("Accuracy: ", accuracy_result)

# Based on lectures and practical Jupyter Notebooks
# This function performs a k-fold evaluation of the model by splitting the data into k-portions, training/testing appropriately with each,
# and determining the average accuracy.
# While in most circumstances this would be a good approach, the size of the dataset provided is extremely limiting
# Splitting the dataset into 10 sets results in roughly 1150 "neither" data, 197 "racism" data and 343 "sexism" data.
# This isn't quite enough to successfully train the algorithm.
# It was implemented for comparison purposes, but for the more successful algorithm see "train_predict_count"
def evaluate_model(data_words, data_combined, tested_dataset, stop_words, folds):
	precision = []
	recall = []
	f1 = []
	accuracy = []

	reports = []

	print("Generating " + str(folds) + "-Fold Evaluation...")
	warnings.filterwarnings("ignore")

	# This approach splits the dataset into k portions
	# Therefore, unlike with train_predict_count, the data needn't be shuffled every iteration
	# As with last time - combine_percentage cross-combines the three tweet lists and gives them appropriate labels while
	# split separates the resultant data into training and testing datasets
	combine_percentage(data_words, data_combined, 50, 100, 100)
	split(data_combined)

	# In the first program, I used my own version of this. I preferred the way I wrote the code as the data isn't perfect and so I could make up for it
	# Here, however, the writers explicitely stated they used sklearn - thus, it is assumed they would have used sklearn's own train_test_split function
	for i in range(folds):
		print("[" + str(i + 1) + " / " + str(folds) + "]")
		# Take the training and testing datasets and extract the i'th portion from it
		# x_train = training data
		# x_test = testing data
		# y_train = training labels
		# y_test = testing labels
		x_train, x_test, y_train, y_test = create_test_train_section(data_combined[tested_dataset], i, folds)

		# The TfidVectoriser combines the CountVectoriser and TfidTransformer
		# "We consider unigram, bigram and trigram features" :: ngram_range = (1, 3)
		# Whether character or word n-grams are used isn't defined, however "the TFIDF approach on the bagof-words features also show promising results" suggests word n-ngrams :: analyzer = "word"
		vectorizer = TfidfVectorizer(min_df=0, sublinear_tf=True)
		data_combined[tested_dataset]["training"]["vectors"] = vectorizer.fit_transform(x_train)
		data_combined[tested_dataset]["test"]["vectors"] = vectorizer.transform(x_test)
		
		# The classifier can use stop_words. These aren't currently used as they, on average, lower accuracy
		#classifier = svm.SVC(kernel='linear', gamma="scale", stop_words=stop_words)
		classifier = svm.SVC(kernel='linear', gamma="scale")
		classifier.fit(data_combined[tested_dataset]["training"]["vectors"], y_train)
		prediction = classifier.predict(data_combined[tested_dataset]["test"]["vectors"])

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

# This project takes 3 json datasets provided in the module, "racism", "sexism" and "neither" which store a list of tweets with either racist, sexist, or generally hateful messages
# which are used to train and test a Machine Learning algorithm for determining the category of a tweet.
def main():
	print("generating data...")
	# First, the files are loaded into three lists - "racism.json" gets loaded into "racism", "sexism.json" gets loaded into "sexism," and "neither.json" is loaded into "neutral."
	load(words)

	# Next, a list of manually selected stop-words are loaded in. These were selected manually because of the specific context - words like "he" or "she" are considered stop-words, but
	# they can carry identifiable meaning in this case. Thus, these words have been removed from the stop-words list
	stop_words = load_list("stopwords.txt")

	# The datasets are combined into three new sets:
	#	 combined["racism"] contains a combination of words["neutral"] and words["racism"]
	#	 combined["sexism"] contains a combination of words["neutral"] and words["sexism"]
	#	 combined["all"] contains a combination of words["neutral"], words["racism"] and words["sexism"]
	# This variable determines which dataset is to be used in evaluation.
	tested_dataset = "all"
	repetitions = 10

	print("generating [" + str(repetitions) + "] predictions...")

	# There is a problem with the data: The dataset isn't very large. The standard evaluation technique breaks the dataset into 10 portions; this is problematic when the dataset is so small.
	# Thus, two evaluation approaches were developed for the model.
	# "evaluate_model" uses the approach described in the practical - the dataset is split into 10 and used to run 10 evaluations the results of which are then averaged.
	# "train_predict_count" instead uses the full dataset every time but shuffled.
	# Each is described in more detail in their respective function.

	#evaluate_model(words, combined, tested_dataset, stop_words, repetitions)
	train_predict_count(words, combined, tested_dataset, stop_words, repetitions)

main()
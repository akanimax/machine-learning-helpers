import collections # only dependency for this function

'''
	Note: this function assumes the input as a list of words in a meaningful sequence. This function can be easily modified for handling list of sequences as input for special cases of seq2seq models.
'''

def build_dataset(words, n_words):
	"""Process raw inputs into a dataset."""
 	count = [['UNK', -1]] # start with this list.
 	count.extend(collections.Counter(words).most_common(n_words - 1)) # this is inplace. i.e. has a side effect

	dictionary = dict() # initialize the dictionary to empty one
	# fill this dictionary with the most frequent words
	for word, _ in count:
		dictionary[word] = len(dictionary)
  
	# loop to replace all the rare words by the UNK token
	data = list() # start with empty list
	unk_count = 0 # counter for keeping track of the unknown words
	for word in words:
		if word in dictionary:
			index = dictionary[word]
		else:
			index = 0  # dictionary['UNK']
			unk_count += 1
		data.append(index)
  
	count[0][1] = unk_count # replace the earlier -1 by the so calculated unknown count

 	print("Total rare words replaced: ", unk_count) # log the total replaced rare words
  
	# construct the reverse dictionary for the original dictionary
	reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))

	# return all the relevant stuff	
	return data, count, dictionary, reversed_dictionary

import numpy as np
import copy

# Given a list of predictions in the form of softmax probabilities, generate a list 
# of lists in the form of [n*** label string, english words label]
def translate_predictions(predictions, labels_indexes, labels_english):
	# swap labels_indexes (key becomes value, value becomes key)
	swap_labels_indexes = copy.deepcopy(labels_indexes)
	swap_labels_indexes = {v: k for k, v in labels_indexes.items()}

	result = []
	for prediction in predictions:
		index = np.argmax(prediction, axis=0)
		n_prefix = swap_labels_indexes[index]
		result.append((n_prefix, labels_english[n_prefix]))

	return result
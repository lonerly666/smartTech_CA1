import numpy as np
import copy

# Given a list of predictions in the form of softmax probabilities, generate a list 
# of lists in the form of [image_name, n*** label string, english words label]

# only use test_name_indexes when running the whole test dataset.
def translate_predictions(predictions, labels_indexes, labels_english, test_names_indexes=None):
    # swap labels_indexes (key becomes value, value becomes key)
    swap_labels_indexes = copy.deepcopy(labels_indexes)
    swap_labels_indexes = {v: k for k, v in labels_indexes.items()}

    result = []
    for i, prediction in enumerate(predictions):
        index = np.argmax(prediction, axis=0)
        n_prefix = swap_labels_indexes[index]
        result.append((
            None if test_names_indexes is None else test_names_indexes[i],
            n_prefix, 
            labels_english[n_prefix]
        ))

    return result
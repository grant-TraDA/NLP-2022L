
from sklearn.metrics import f1_score

def calculate_stats(test_y, predictions):
    """
    Calculates statistcs for given true values and predictions
    Arguments:
        - test_y: list of ints (true values)
        - predictions: list of ints (predicted values)
    Returns:
        - dictionary of calculated statistics (string -> float)
    """
    micro_fscore = f1_score(test_y, predictions, average='micro')
    macro_fscore = f1_score(test_y, predictions, average='macro')
    return { "Micro F1-score": micro_fscore, "Macro F1-score": macro_fscore }

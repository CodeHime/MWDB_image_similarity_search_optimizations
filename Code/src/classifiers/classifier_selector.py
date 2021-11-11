from svm import *
from ppr import *
from decision_tree import *


def classify_data(data, classifier="svm"):
    if classifier=="svm":
        raise NotImplmentedError(f"No implementation found for selected classifier: {classifier}")
    elif classifier=="ppr":
        raise NotImplmentedError(f"No implementation found for selected classifier: {classifier}")
    elif classifier=="decision_tree":
        raise NotImplmentedError(f"No implementation found for selected classifier: {classifier}")
    else:
        raise NotImplmentedError(f"No implementation found for selected classifier: {classifier}")
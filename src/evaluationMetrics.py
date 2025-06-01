def topKAcc(actualResult, predictedResult, K=1):
    if not actualResult or not predictedResult:
        return 0.0
    correct = 0
    for i in range(min(K, len(predictedResult))):
        if predictedResult[i] in actualResult:
            correct += 1
    return correct / min(K, len(actualResult))

def NDCG(actualResult, predictedResult):
    def DCG(scores):
        from math import log
        return sum((2 ** score - 1) / (log(i + 2) / log(2)) for i, score in enumerate(scores) if score > 0)
    if not actualResult or not predictedResult or len(actualResult) != len(predictedResult):
        return 0.0
    ideal_dcg = DCG(sorted(actualResult, reverse=True))
    if ideal_dcg == 0:
        return 0.0
    pred_dcg = DCG(predictedResult)
    return pred_dcg / ideal_dcg

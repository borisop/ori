from src.svm import SVM

svm_pos = SVM()
svm_neg = SVM()

svm_pos = SVM.load('SVM_sentiment')
svm_neg = SVM.load('SVM_toxicity_level')

print('\nSentiment classification')
svm_pos.print_stats()
print('\nToxicity level classification')
svm_neg.print_stats()

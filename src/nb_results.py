from src.nb import NB

nb_pos = NB()
nb_neg = NB()

nb_pos = NB.load('NB_sentiment')
nb_neg = NB.load('NB_toxicity_level')

print('\nSentiment classification')
nb_pos.print_stats()
print('\nToxicity level classification')
nb_neg.print_stats()

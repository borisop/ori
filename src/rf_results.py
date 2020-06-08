from src.rf import RF

rf_pos = RF()
rf_neg = RF()

rf_pos = RF.load('RF_sentiment')
rf_neg = RF.load('RF_toxicity_level')

print('\nSentiment classification')
rf_pos.print_stats()
print('\nToxicity level classification')
rf_neg.print_stats()

from src.xgb import XGB

xgb_pos = XGB()
xgb_neg = XGB()

xgb_pos = XGB.load('XGB_sentiment_new')
xgb_neg = XGB.load('XGB_toxicity_level_new')

print('\nSentiment classification')
xgb_pos.print_stats()
print('\nToxicity level classification')
xgb_neg.print_stats()

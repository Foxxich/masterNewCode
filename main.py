from hybrid_voting_boosting import HybridVotingBoosting
from weighted_consensus import WeightedConsensus
from adaptive_feature_selection import AdaptiveFeatureSelectionBoosting
from random_decision_tree_boosting import RandomDecisionTreeBoosting
import pandas as pd

train_file = r'C:\Users\Vadym\Documents\masterNewCode\train.csv'
test_file = r'C:\Users\Vadym\Documents\masterNewCode\test.csv'
submit_file = r'C:\Users\Vadym\Documents\masterNewCode\submit.csv'

# uruchamiamy Hybrid Voting Boosting
hv_boost = HybridVotingBoosting(train_file, test_file, submit_file)
predictions = hv_boost.fit_predict()
hv_boost.save_predictions(predictions, 'submit_hybrid_voting.csv')

# obliczenia accuracy dla Hybrid Voting Boosting
hv_boost.evaluate('submit_hybrid_voting.csv', submit_file)

# uruchamiamy Weighted Consensus
wc_boost = WeightedConsensus(train_file, test_file, submit_file)
predictions = wc_boost.fit_predict()
wc_boost.save_predictions(predictions, 'submit_weighted_consensus.csv')

# obliczenia accuracy dla Weighted Consensus
wc_boost.evaluate('submit_weighted_consensus.csv', submit_file)

# uruchamiamy Self-Adaptive Feature Selection Boosting
afs_boost = AdaptiveFeatureSelectionBoosting(train_file, test_file, submit_file)
predictions = afs_boost.fit_predict()
afs_boost.save_predictions(predictions, 'submit_adaptive_feature.csv')

# obliczenia accuracy dla Adaptive Feature Selection Boosting
afs_boost.evaluate('submit_adaptive_feature.csv', submit_file)

# uruchamiamy Random Decision Tree Boosting
rd_boost = RandomDecisionTreeBoosting(train_file, test_file, submit_file)
predictions = rd_boost.fit_predict()
rd_boost.save_predictions(predictions, 'submit_random_decision_tree.csv')

# obliczenia accuracy dla Random Decision Tree Boosting
rd_boost.evaluate('submit_random_decision_tree.csv', submit_file)

# Predicting NBA Game Outcomes and Identifying Mispriced Contracts in Sports Prediction Markets Using Machine Learning

**Authors:** Noah Arooji (nun2as), Austin Vu (tcu3wh), Azim Abdulmajeeth (zwf8qy), Ciaran Jones (gff7en)

## 1. Problem Statement
Sports prediction markets, such as Kalshi and Polymarket, allow participants to trade contracts on the outcomes of real-world events, including professional basketball games. These platforms operate similarly to financial exchanges. Each contract is priced between $0 and $1, and the price reflects the market’s implied probability of an outcome occurring. For example, a contract priced at $0.65 for a team to win implies the market believes that team has a 65% chance of winning. As these markets grow in volume and popularity, they create an opportunity to apply machine learning to detect mispriced contracts, meaning cases where a model’s estimated probability differs meaningfully from the market-implied probability.

Traditional sports betting often relies on expert intuition or simple statistical models. While sportsbooks use advanced internal models to set efficient lines, prediction market prices are set by participants and may adjust more slowly to new information such as rest days, recent performance trends, or matchup-specific factors. This project aims to take advantage of potential inefficiencies by training supervised learning models to predict the true probability of NBA game outcomes, including moneyline winners and over/under totals. These predictions will then be compared to contract prices on Kalshi and Polymarket.

The model inputs will include features based on team and individual statistics, and current market prices. The output will be a predicted probability for each game outcome, so that we can flag probabilities that vary significantly from the market’s implied probability.

## 2. Literature Review
The application of machine learning in sports outcome prediction has been well-studied. Researchers have applied logistic regression, support vector machines, and neural networks to predict outcomes in the NBA with varying degrees of success, generally achieving classification accuracies in the range of 65–72% depending on feature engineering and model complexity [1, 2]. A common finding across the literature is that ensemble methods, particularly gradient-boosted decision trees, will outperform individual classifiers in this domain because of their ability to capture interactions between features such as pace, rest days, and home-court advantage [3].

Prediction markets have also received significant attention in the economics and finance literature. Studies have shown that prediction market prices are generally well-calibrated, so a contract priced at $0.70 wins approximately 70% of the time [4]. However, researchers have also identified systematic inefficiencies, including the favorite-longshot bias and slow adjustment to late-breaking information, which create opportunities for informed trading [5]. Recent work has attempted to combine machine learning predictions with market prices to generate profitable trading strategies in both sports and political prediction markets [6].

This project draws on influences from both by using standard classification models from the machine learning curriculum to generate probability estimates, and then evaluating those estimates not only on predictive accuracy but also on simulated profitability when used to trade against prediction market prices.

## 3. Data Collection
All data for this project will be collected from free, publicly available sources. Team-level NBA statistics, including offensive and defensive ratings, pace, recent win-loss records, and home/away splits, will be sourced from the `nba_api` Python package, an open-source client that pulls directly from the official NBA.com stats endpoints at no cost. Historical and real-time prediction market contract prices will be collected from the Kalshi API, which provides free access to both current and historical contract pricing for NBA moneyline and over/under markets. Because Kalshi’s historical data includes timestamped contract prices for past games, it will serve as the primary source for market-implied probabilities in both the training set and the evaluation backtest.

Polymarket data will be collected where available via their public API to supplement the Kalshi data. Additionally, The Odds API’s free tier may be used to pull current-day sportsbook lines as a supplementary feature when available, though it will not serve as a primary historical data source.

The core feature set will be constructed by joining game-level data from `nba_api` with the corresponding Kalshi market data. Features will include team offensive and defensive ratings, pace of play, number of rest days since the previous game, home/away indicator, rolling averages over the last N games for important statistics, and the Kalshi contract price (market-implied probability). The target variable will be binary for moneyline prediction (win/loss) and binary for over/under prediction (over/under the posted total).

## 4. Algorithms and Models
We will implement several supervised learning models covered in this course, selecting those best suited for the problem.

### 4.1 Logistic Regression
Logistic regression serves as the baseline for this binary classification task. It is highly interpretable, allowing us to inspect the learned coefficients and understand which features (e.g., rest days, defensive rating differential) contribute most to the predicted outcome. We will use L2 regularization to prevent overfitting given the moderate number of features relative to the number of games per season.

### 4.2 Ensemble Methods
Ensemble methods, specifically Random Forests and Gradient-Boosted Decision Trees (e.g., XGBoost), are the primary models we expect to deliver the strongest predictive performance. Random Forests are robust to overfitting and naturally handle feature interactions without extensive feature engineering. Gradient boosting, on the other hand, builds trees sequentially to correct prior errors and has been shown to excel in structured tabular prediction tasks. These models also provide feature importance scores, which will help identify whether the market price itself, team performance metrics, or scheduling factors carry the most predictive weight. We will also tune key hyperparameters such as the number of estimators, maximum tree depth, and learning rate.

## 5. Evaluation
Model evaluation will be conducted along two dimensions: predictive quality and economic value. For predictive quality, we will use accuracy (overall classification correctness), log-loss (which penalizes confident but incorrect predictions), and Brier score (the mean squared error of predicted probabilities against binary outcomes). Log-loss and Brier score are especially important in this project because we are not simply classifying winners and losers, we are estimating probabilities that must be well-calibrated to be useful for identifying mispriced contracts.

For economic value, we will conduct a simulated trading test. The strategy is straightforward: when the model’s predicted probability diverges from the Kalshi/Polymarket probability by more than a certain threshold, a simulated trade is placed. We will measure the return on investment (ROI) and the overall profit/loss of this strategy across the test set. Baseline comparisons will include a naive model that always predicts the home team to win, a model that uses only the Kalshi contract price as its prediction, or logistic regression as the simplest learned model.

## 6. Timeline
The project will be completed over approximately seven weeks. During the first two weeks, we will complete data collection from the `nba_api` package and the Kalshi API, and define our feature sets. Weeks three and four will be for implementing and training the core models: logistic regression and ensemble methods. In week five, we will perform hyperparameter tuning and evaluate all models on the test set. Week six will focus on the simulated trading test, where we will measure profitability across different thresholds. The final week will be dedicated to preparing the project report and presentation.

## 7. Expected Results
We expect that ensemble methods, especially gradient-boosted decision trees, will have the strongest predictive accuracy and best-calibrated probabilities, consistent with findings in the literature on tabular prediction tasks. Logistic regression should serve as a competitive and interpretable baseline, and the comparison between these two model families will help characterize whether the nonlinear feature interactions captured by tree-based ensembles provide meaningful lift over a linear model in this domain. We anticipate overall classification accuracy in the range of 65–70%, which is consistent with prior work on NBA game prediction.

More importantly, we expect to identify a subset of games where the model’s predicted probability diverges meaningfully from the Kalshi contract price, and we hypothesize that a simple threshold-based trading strategy will yield positive simulated ROI on these contracts. Even if the absolute accuracy improvement over the market is modest, the project will demonstrate the full pipeline of training a probabilistic classifier, calibrating its outputs, and applying it to a real-world decision-making context in prediction markets.

---
**References:**
1. F. Thabtah, L. Zhang, and N. Abdelhamid. 2019. NBA Game Result Prediction Using Feature Analysis and Machine Learning. *Annals of Data Science*, 6(1), 103–116.
2. B. Loeffelholz, E. Bednar, and K. Bauer. 2009. Predicting NBA Games Using Neural Networks. *Journal of Quantitative Analysis in Sports*, 5(1).
3. T. Chen and C. Guestrin. 2016. XGBoost: A Scalable Tree Boosting System. *In Proc. 22nd ACM SIGKDD*, 785–794.
4. J. Wolfers and E. Zitzewitz. 2004. Prediction Markets. *Journal of Economic Perspectives*,
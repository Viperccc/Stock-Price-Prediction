# Stock-Price-Prediction
# Stock Price Prediction

A curated list of deep learning stock price prediction since 2017



## Background

Alternative data for stock price prediction. I believe stock price prediction is very important for the fusion of finance and deep learning, or it is the future of the finance. So before we start to do the research of stock price prediction, we need to know the related work and some classical papers for stock price prediction. What is the data type and the common used model types (e.g. LSTM..). Therefore, I decided to make a repository of a list of deep learning stock price prediction papers and codes to help others. Hope we can cotinue to do this year by year.



## Paper&Code



#### **Stock Price Prediction via Discovering Multi-Frequency Trading Patterns**

- Conference: KDD2017
- Pdf: https://dl.acm.org/doi/10.1145/3097983.3098117
- Dataset: Yahoo! Finance (50 stocks among 10 sectors from 2007 to 2016)
- Abstract:  No related works explicitly decompose trading patterns into various frequency components, and seamlessly integrate the discovered multi-frequency patterns into price predictions.  This inspires us to discover and leverage the trading patterns of multiple frequencies. We present the SFM recurrent network that learns multiple frequency trading patterns as to make short and long-term price predictions.
- Contribution: Model; Discrete Fourier Transform; 



#### DP-LSTM: Differential Privacy-inspired LSTM for Stock Prediction Using Financial News

- Conference: NeurIPS2019
- Pdf: https://arxiv.org/pdf/1912.10806v1.pdf
- Dataset: Yahoo Finance (12/07/2017 to 06/01/2018) + news article from financial domain are collected with the same time period as stock data.
- Abstract:  In this paper, we use sentiment analysis to extract information from textual information. Sentiment analysis is the automated process of understanding an opinion about a given subject from news articles. We proposed a DP-LSTM neural network, which increase the accuracy of prediction and robustness of model at the same time. Using the informantion of articals by predefined the postive words or negative words for each stock.
- Contribution: External information (news) 



#### Deep learning for financial applications : A survey

- Journal: Applied Soft Computing
- Pdf: https://www.sciencedirect.com/science/article/pii/S1568494620303240?via%3Dihub
- Dataset: xxx
- Abstract: A survey



#### Temporal Relational Ranking for Stock Prediction

- Journal: TOIS

- Pdf: https://dl.acm.org/doi/pdf/10.1145/3309547

- Dataset: NYSE& NASDAQ (We collect the stocks from the NASDAQ and NYSE markets that have transaction records between 01/02/2013 and 12/08/2017, obtaining 3,274 and 3,163 stocks, respectively)

- Abstract: Related works typically formulate stock prediction as a classification (to predict stock trends) or a regression problem (to predict stock prices). They treat the stocks as independent of each other, which degrades the model performance. In this paper, we propose a new deep learning solution, named Relational Stock Ranking (RSR). Specifically, we propose of a new component in neural network modeling, named Temporal Graph Convolution, which jointly models the temporal evolution and relation network of stocks.

  -If two companies are in the same sector or industry, they may exhibit similar trends in their stock prices,

  -If two companies are partners in a supply chain, then the events of the upstream company may affect the stock price of the downstream company

  Given the multi-hot binary relation encodings extracted by the rule, we use GCN to do the Temporal Graph Convolution.

- Contribution: Consider the relationships between different stocks and use the GCN to learn the embeddings

  

#### Enhancing Stock Movement Prediction with Adversarial Training 

- Conference: IJCAI2019
- Pdf: https://arxiv.org/pdf/1810.09936.pdf
- Dataset: ACL18 (contains historical data from Jan-01-2014 to Jan01-2016 of 88 high-trade-volume-stocks in NASDAQ and NYSE markets) & KDD17 (includes longer history ranging from Jan-01-2007 to Jan-01-2016 of 50 stocks in U.S. markets)
- Abstract: The key novelty is that we propose to employ adversarial training to improve the generalization of a neural network prediction model. The rationality of adversarial training here is that the input features to stock prediction are typically based on stock price, which is essentially a stochastic variable and continuously changed with time by nature.
- Contribution: Using adversarial training to improve the generalization of a neural network prediction model.



#### Knowledge-Driven Stock Trend Prediction and Explanation via Temporal Convolutional Network

- Conference: WWW2019
- Pdf: https://dl.acm.org/doi/pdf/10.1145/3308560.3317701
- Dataset: Yahoo Finance (8/08/2008 to 01/01/2016.)
- Abstract: Existing works have two shortcomings: 1) not sensitive enough to abrupt changes of stock trend 2) forecasting results are not interpretable for humans. To address the problems, we combine event embeddings and price values together to forecast stock trend. We evaluate the prediction accuracy to show how knowledge-driven events work on abrupt changes. Use OpenIE to represent the events. After getting concise event tuples, we construct a sub-graph from KG by utilizing the technique of entity linking, in order to disambiguate named entities in texts by associating them with predefined entities in KG.  After we have the event embeddings, we concate it with price and then feed into Temporal Convolutional Network for training.
- Contribution: Using and combining additional information: events



#### Generating Realistic Stock Market Order Streams

- Conference: AAAI2020
- Pdf: https://ojs.aaai.org/index.php/AAAI/article/view/5415
- Dataset: OneMarketData,
- Abstract: We propose an approach to generate realistic and highfidelity stock market data based on generative adversarial networks (GANs). We utilize a conditional Wasserstein GAN to capture the time-dependence of order streams and propose a a mathematical characterization of the distribution learned by the generator.
- Contribution: Using GAN to generate the high-quality stock market data.



#### Neural Networks Fail to Learn Periodic Functions and How to Fix It

- Conference: NeurIPS2020
- Pdf: https://proceedings.neurips.cc/paper/2020/file/1160453108d3e537255e9f7b931f4e90-Paper.pdf
- Dataset: Wilshire 5000 Total Market Full Cap Index (US market)
- Abstract: We prove and demonstrate experimentally that the standard activations functions, such as ReLU, tanh, sigmoid, along with their variants, all fail to learn to extrapolate simple periodic functions. We hypothesize that this is due to their lack of a “periodic” inductive bias. As a fix of this problem, we propose a new activation, namely, x + sin^2(x), which achieves the desired periodic inductive bias to learn a periodic function while maintaining a favorable optimization property of the ReLU-based activations. The global economy is another area where quasi-periodic behaviors might happen. At microscopic level, the economy oscillates in a complex, unpredictable manner; at macroscopic level, the global economy follows a 8−10 year cycle that transitions between periods of growth and recession.
- Contribution: Propose a novel avtivations functions to solve the periodic data.



#### REST: Relational Event-driven Stock Trend Forecasting

- Conference: WWW2021
- Pdf: https://dl.acm.org/doi/abs/10.1145/3442381.3450032
- Dataset: CSI 300 and CSI 500
- Abstract: Many event-driven methods utilized the events extracted from news, social media, and discussion board to forecast the stock trend in recent years. However, existing event-driven methods have two main shortcomings: 1) overlooking the influence of event information differentiated by the stock-dependent properties; 2) neglecting the effect of event information from other related stocks.To remedy the first shortcoming, we propose to model the stock context and learn the effect of event information on the stocks under different contexts. To address the second shortcoming, we construct a stock graph and design a new propagation layer to propagate the effect of event information from related stocks. We use the GCN layer to propagate event information on the stock graphs. Specifically, , we design a new propagation layer to model the complex cross-stock influence inspired by a couple of characteristics observed in the real-world.
- Contribution: Propose a novel GCN with propagation layer to capture the complex cross-stock influence between different stocks. 
  - Different relations has different effects (Different types of edges)
  - The weight is dynamically changed
  - Capture multi-hop influence of events



#### Stock Selection via Spatiotemporal Hypergraph Attention Network: A Learning to Rank Approach

- Conference: AAAI2021
- Pdf: https://ojs.aaai.org/index.php/AAAI/article/view/16127
- Dataset: NASDAQ
- Abstract: Existing methods face two significant limitations. 1) They do not directly optimize the target of investment in terms of profit 2) They treat each stock as independent from the others, ignoring the rich signals between related stocks’ temporal price movements. Building on these limitations, we reformulate stock prediction as a learning to rank problem and propose STHAN-SR, a neural hypergraph architecture for stock selection.
- Contribution:  Use ranking loss to train the model






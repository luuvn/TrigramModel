# TrigramModel

============
Introduction
============

TrigramModel process brown corpus, reuters corpus and calculates trigram
probabilities according to the Katz back-off method.

============
How to use
============

1. 2 folders brown_reformat and reuters_reformat contains files that're already formatted
as each sentence per line.

2. run python main.js:
  - first time, it will build 2 trigram model (brown_train, reuters_train) base on 2 folder of corpus file above.
  - second time, you can choose the way to interact with these trigram model:

== Enter your choice:
  1.load trigram model
  2.test perplexity
  3.generate random sentence

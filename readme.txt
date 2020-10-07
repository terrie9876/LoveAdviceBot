Project Name: Love Advice Bot
Author: Terry Shih

Language: Python 3.5
Required Packages for running: markovify, gensim, pickle, json
Additional packages for building project: praw, contractions, nltk, numpy, re, num2words

How to Run:
Put your story/question into Input.txt that's located in the I-O folder. Run AdviceGenerator.py. Once that's done, the "advice" should show up in Output.txt that's also located in the I-O folder.

How to Build:
Ensure that the Raw Data folder is empty. Once you have the requisite libaries, run the .py files in the following order:
-DataCollector.py
-ModelMaker.py
-MarkovMaker.py
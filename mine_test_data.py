import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

target_names = ['Politics','Sports']
f = open('En_training.txt','r')
data = []
target = []
tweetid = []
for line in f:
	l = []
	l = line.split()
	tweetid.append(l[0])
	data.append(' '.join(l[2:]))
	if l[1] == 'Politics':
		target.append(0)
	elif l[1] == 'Sports':
		target.append(1)

f = open('ValidationOutput.txt','r')
data = []
target = []
tweetid = []
for line in f:
	l = []
	l = line.split()
	tweetid.append(l[0])
	data.append(' '.join(l[2:]))
	if l[1] == 'Politics':
		target.append(0)
	elif l[1] == 'Sports':
		target.append(1)

f = open('En_test.txt','r')
test_data = []
test_tweetid = []
for line in f:
  l = []
  l = line.split()
  test_tweetid.append(l[0])
  test_data.append(' '.join(l[2:]))

classifier = Pipeline([
		('vectorizer', CountVectorizer()),
		('tfidf', TfidfTransformer()),
    ('clf', MultinomialNB(alpha=0.1,fit_prior=True))])

classifier.fit(data,target)
predicted =  classifier.predict(test_data)
with open('TestOutput.txt', 'w+') as opfid:
	for i in range(len(test_data)):
		if predicted[i] == 0:
			opfid.write(test_tweetid[i]+' '+'Politics')
		elif predicted[i] == 1:
			opfid.write(test_tweetid[i]+' '+'Sports')
		opfid.write('\n')	

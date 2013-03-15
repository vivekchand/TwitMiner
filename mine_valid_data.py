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

f = open('En_validation.txt','r')
valid_data = []
valid_tweetid = []
for line in f:
  l = []
  l = line.split()
  valid_tweetid.append(l[0])
  valid_data.append(' '.join(l[2:]))

classifier = Pipeline([
		('vectorizer', CountVectorizer()),
		('tfidf', TfidfTransformer()),
    ('clf', MultinomialNB(alpha=1.4,fit_prior=True))
	])

classifier.fit(data,target)
predicted =  classifier.predict(valid_data)
with open('ValidationOutput.txt', 'w+') as opfid:
	for i in range(len(valid_data)):
		if predicted[i] == 0:
			opfid.write(valid_tweetid[i]+' '+'Politics'+' '+'"'+valid_data[i]+'"')
#			opfid.write(valid_tweetid[i]+' '+'Politics')
		elif predicted[i] == 1:
			opfid.write(valid_tweetid[i]+' '+'Sports'+' '+'"'+valid_data[i]+'"')	
#			opfid.write(valid_tweetid[i]+' '+'Sports')	
		opfid.write('\n')



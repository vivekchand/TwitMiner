import re

#start process_tweet
def processTweet(tweet):
    # process the tweets

    #Convert to lower case
    tweet = tweet.lower()
    #Convert www.* or https?://* to URL
    tweet = re.sub('((www\.[\s]+)|(https?://[^\s]+))','URL',tweet)
    #Convert @username to AT_USER
    tweet = re.sub('@[^\s]+','AT_USER',tweet)
    #Remove additional white spaces
    tweet = re.sub('[\s]+', ' ', tweet)
    #Replace #word with word
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
    #trim
    tweet = tweet.strip('\'"')
    return tweet
#end

#initialize stopWords
stopWords = []

#start replaceTwoOrMore
def replaceTwoOrMore(s):
    #look for 2 or more repetitions of character and replace with the character itself
    pattern = re.compile(r"(.)\1{1,}", re.DOTALL)
    return pattern.sub(r"\1\1", s)
#end

#start getStopWordList
def getStopWordList():
    #read the stopwords file and build a list
    stopWords = []
    stopWords.append('AT_USER')
    stopWords.append('URL')

    fp = open('stopwords.txt', 'r')
    line = fp.readline()
    while line:
        word = line.strip()
        stopWords.append(word)
        line = fp.readline()
    fp.close()
    return stopWords
#end

#start getfeatureVector
def getFeatureVector(tweet):
    featureVector = []
    global train
    #split tweet into words
    words = tweet.split()
    for w in words:
        #replace two or more with two occurrences
        w = replaceTwoOrMore(w)
        #strip punctuation
        w = w.strip('\'"?,.')
        #check if the word stats with an alphabet
        val = re.search(r"^[a-zA-Z][a-zA-Z0-9]*$", w)
        #ignore if it is a stop word
        if(w in stopWords or val is None):
            continue
        else:
            featureVector.append(w.lower())
    return featureVector
#end

stopWords = getStopWordList()
	

target_names = ['Politics','Sports']
f = open('training.txt','r')
data = []
target = []
tweetid = []
with open('En_training.txt', 'w+') as opfid:
	for line in f:
		l = []
		l = line.split()
		tweetid = l[0]
		sentiment = l[1]
		processedTweet = processTweet(' '.join(l[2:]))
		featureVector = getFeatureVector(processedTweet)
		data = '"'+' '.join(featureVector)+'"'
		opfid.write(tweetid+' '+sentiment+' '+data)
		opfid.write('\n')

f = open('validation.txt','r')
data = []
target = []
tweetid = []
with open('En_validation.txt', 'w+') as opfid:
  for line in f:
    l = []
    l = line.split()
    tweetid = l[0]
    processedTweet = processTweet(' '.join(l[1:]))
    featureVector = getFeatureVector(processedTweet)
    data = '"'+' '.join(featureVector)+'"'
    opfid.write(tweetid+' '+data)
    opfid.write('\n')

f = open('test.txt','r')
data = []
target = []
tweetid = []
with open('En_test.txt', 'w+') as opfid:
  for line in f:
    l = []
    l = line.split()
    tweetid = l[0]
    processedTweet = processTweet(' '.join(l[1:]))
    featureVector = getFeatureVector(processedTweet)
    data = '"'+' '.join(featureVector)+'"'
    opfid.write(tweetid+' '+data)
    opfid.write('\n')



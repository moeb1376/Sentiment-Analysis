import json
from afinn import Afinn
import random
from sklearn.metrics import precision_recall_fscore_support
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import matplotlib.pyplot as plt
import csv

not_list = ['not', "isn't", "doesn't", "didn't", "don't", "wouldn't", "couldn't", "shouldn't"]
year = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
year_history = [[], [], [], [], [], [], []]

for y in year_history:
    for i in range(12):
        y.append([0, 1])
    y.append([0, 1])

ps = PorterStemmer()
data = []
with open("elonmusk_tweets.csv", 'r') as f:
    spamreader = csv.reader(f, delimiter=',', quotechar='|')
    c = 0
    for row in spamreader:
        if c == 0:
            c += 1
            continue
        s = ' '.join(row[2:])
        b = s[1:len(s) - 1]
        text = b[2:len(b) - 1]
        date = row[1].split(' ')[0].split('-')
        y = int(date[0])
        m = int(date[1])
        c += 1
        data.append({'text': text, 'created_at': (y, m)})


def afinn():
    with open("AFINN-111.txt", 'r', encoding='UTF-8') as f:
        lines = f.readlines()
        pn = {}
        for l in lines:
            w = l.split('\t')
            pn[ps.stem(w[0])] = w[1][:-1]

    return pn


def afinn_sentiment(tweet, polarity):
    tweet = word_tokenize(tweet)
    sentiment = 0

    for i in range(len(tweet)):
        tweet[i] = tweet[i].lower()

    for i in range(len(tweet)):
        if tweet[i] in polarity:
            is_not = check_not(tweet, i)
            if is_not:
                sentiment -= int(polarity[tweet[i]])
            else:
                sentiment += int(polarity[tweet[i]])
            if int(polarity[tweet[i]]) <= 0:
                sentiment += 1.5 * int(polarity[tweet[i]])

    return sentiment


def check_not(text, position):
    for i in range(position - 3, position):
        if text[i] in not_list:
            return True
    return False


def get_sample(src_list, sample_list):
    for i in range(100):
        r = random.randint(1, len(src_list))
        sample_list.append(src_list.pop(r))
    return sample_list


def final_normalize(lst):
    lst.reverse()
    for i in range(4):
        lst.pop()
    lst.reverse()
    for i in range(5):
        lst.pop()


pn = afinn()

sample = []
sample = get_sample(data, sample)

total_history = []

for d in data:
    sent = afinn_sentiment(d['text'], pn)
    total_history.append((d['created_at'], sent))

    year_history[d['created_at'][0] - 2011][d['created_at'][1]][0] += sent
    year_history[d['created_at'][0] - 2011][d['created_at'][1]][1] += 1
    year_history[d['created_at'][0] - 2011][12][0] += sent
    year_history[d['created_at'][0] - 2011][12][1] += 1

final = []
for y in year_history:
    for m in range(12):
        final.append(y[m][0] / y[m][1])

# final_normalize(final)

classifier = Afinn(language='en')
real = []
pred = []
for item in sample:
    if classifier.score(item['text']) >= 0:
        real.append('pos')
    else:
        real.append('neg')
    if afinn_sentiment(item['text'], pn) >= 0:
        pred.append('pos')
    else:
        pred.append('neg')

accuracy = precision_recall_fscore_support(real, pred)

print('Precision:\t', accuracy[0])
print('Recall:\t\t', accuracy[1])
print('F1-Score:\t', accuracy[2])

color = []
for i in final:
    if i >= 0:
        color.append("green")
    else:
        color.append("red")
print(len(final))
plt.bar([0], [0], color="red", label="Negative")

plt.bar([i for i in range(84)], final, color=color, label="Positive")
plt.xticks([i for i in range(9, 111, 12)], ['2011', '2012', '2013', '2014', '2015', '2016', '2017'])
plt.title("Sentiment of @elonmusk tweets")
plt.grid()
plt.xlabel("Tweets per month")
plt.ylabel("Avg. sentiment")
plt.legend()
plt.show()

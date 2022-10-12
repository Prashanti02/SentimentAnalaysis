import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

plt.style.use('ggplot')

import nltk
#nltk.download('punkt')
#nltk.download('averaged_perceptron_tagger')
#nltk.download('maxent_ne_chunker')
#nltk.download('words')
#nltk.download('vader_lexicon')

df=pd.read_csv('Reviews.csv')
#print(df.head())
df=df.head(500)

#ax= (df['Score'].value_counts().sort_index().plot(kind='bar', title='Count of Reviews by Stars', figsize=(10,5)))
#ax.set_xlabel('Review stars')
#plt.show()

#basic nltk
#example=df['Text'][50]
#print(example)
#tokens= nltk.word_tokenize(example)
#print(tokens)
#tagged= nltk.pos_tag(tokens)
#print(tagged)
#entities = nltk.chunk.ne_chunk(tagged)
#entities.pprint()

from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm import tqdm

#sia = SentimentIntensityAnalyzer()
#print(sia.polarity_scores('I am so happy!'))
#res = {}
#for i, row in tqdm(df.iterrows(), total=len(df)):
#    text = row['Text']
#    myid = row['Id']
#    res[myid] = sia.polarity_scores(text)

#print(res)



#vaders = pd.DataFrame(res).T
#vaders = vaders.reset_index().rename(columns={'index':'Id'})
#vaders = vaders.merge(df, how='left')
#print(vaders)

#ax = sns.barplot(data=vaders, x='Score', y='compound')
#ax.set_title('Compund Score by Amazon Star Review')
#plt.show()

#fig, axs = plt.subplots(1, 3, figsize=(12, 3))
#sns.barplot(data=vaders, x='Score', y='pos', ax=axs[0])
#sns.barplot(data=vaders, x='Score', y='neu', ax=axs[1])
#sns.barplot(data=vaders, x='Score', y='neg', ax=axs[2])
#axs[0].set_title('Positive')
#axs[1].set_title('Neutral')
#axs[2].set_title('Negative')
#plt.tight_layout()
#plt.show()





#vader sentiment scoring : does not account for relations between words

#roberta pretrained model
#from transformers import AutoTokenizer
#from transformers import AutoModelForSequenceClassification
from scipy.special import softmax

#MODEL = f"cardiffnlp/twitter-roberta-base-sentiment"
#tokenizer = AutoTokenizer.from_pretrained(MODEL)
#model = AutoModelForSequenceClassification.from_pretrained(MODEL)

#encoded_text = tokenizer(example, return_tensors='pt')
#print(encoded_text)
#output = model(**encoded_text)
#scores = output[0][0].detach().numpy()
#scores = softmax(scores)
#print(scores)
#scores_dict = {
     #   'roberta_neg' : scores[0],
      #  'roberta_neu' : scores[1],
      #  'roberta_pos' : scores[2]
#}
#return scores_dict




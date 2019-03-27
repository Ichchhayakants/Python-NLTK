from nltk.tokenize import sent_tokenize, word_tokenize

#import nltk
#nltk.download()
'''
example_text = "Hello there, how are you doing today? The weather is great and Python is awesom. The sky is pinkish-blue. You should not eat cardboard."

print(sent_tokenize(example_text))
print(word_tokenize(example_text))


for i in word_tokenize(example_text):
	print(i)
'''

## Stop words
from nltk.corpus import stopwords
example_sentence = "this is an example shownig of stop word filteration"
stop_words = set(stopwords.words("english"))

words = word_tokenize(example_sentence)

'''filtered_sentence=[]

for w in words:
	if w not in stop_words:
		filtered_sentence.append(w)

print(filtered_sentence)
#print(stop_words)'''

filtered_sentence = [w for w in words if not w in stop_words]

print(filtered_sentence)


from nltk.stem import PorterStemmer

ps = PorterStemmer()

example_words = ["python","pythoner","pythoning","pythoning","pythonly"]

'''for w in example_words:
	print(ps.stem(w))
'''

new_text = "It is very important to be pythonly while youn are pythoning with python. All pythoners have pythoned poorly at least once"

words = word_tokenize(new_text)

for w in words:
	print(ps.stem(w))



import nltk
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer


train_text = state_union.raw("2005-GWBush.txt")
sample_text = state_union.raw("2006-GWBush.txt")

custom_sent_tokenizer = PunktSentenceTokenizer(train_text)

tokenized = custom_sent_tokenizer.tokenize(sample_text)

def process_content():
	try:
		for i in tokenized:
			words = nltk.word_tokenize(i)
			tagged = nltk.pos_tag(words)
			'''#print(tagged)
			#chunkGram = r"""Chunk: {<RB.?>*<VB.?><NNP>+<NN>?}"""
			chunkGram = r"""Chunk: {<.*>+}
									}<VB.?|IN|DT>+{"""

			chunkParser = nltk.RegexpParser(chunkGram)
			chunked = chunkParser.parse(tagged)
			print(chunked)
			chunked.draw()'''


			## 7 named entity Recognition
			
			namedEnt = nltk.ne_chunk(tagged,binary=True)
			#namedEnt.draw()




	except Exception as e:
		print(str(e))

process_content()



# Lemmatizing 
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
print(lemmatizer.lemmatize("cats"))
print(lemmatizer.lemmatize("cacti"))
print(lemmatizer.lemmatize("geese"))
print(lemmatizer.lemmatize("rocks"))
print(lemmatizer.lemmatize("python"))


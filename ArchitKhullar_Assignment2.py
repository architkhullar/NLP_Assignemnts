'''
ASSIGNEMNT 2
Problem 1 (Language Model Creation) (80 points). 
In this exercise, you will train probabilistic language models to distinguish between words in different languages. Rather than looking up whole words in a dictionary, you will build models of character sequences so you can make a guess about the language of unseen words. You will need to use NLTK and the Universal Declaration of Human Rights corpus.


We will compare across different languages from the Universal Declaration of Human Rights documents. Use the following code to load the corpus and create sets of four languages.

import nltk 
from nltk.corpus import udhr  

english = udhr.raw('English-Latin1') 
french = udhr.raw('French_Francais-Latin1') 
italian = udhr.raw('Italian_Italiano-Latin1') 
spanish = udhr.raw('Spanish_Espanol-Latin1')  


If you do not have the UDHR dataset already installed with your version of NLTK, use nltk.download() to download the corpus. 

Create training, development and test samples for English, French, Italian, and Spanish from these sets.

english_train, english_dev = english[0:1000], english[1000:1100] 
french_train, french_dev = french[0:1000], french[1000:1100] 
italian_train, italian_dev = italian[0:1000], italian[1000:1100] 
spanish_train, spanish_dev = spanish[0:1000], spanish[1000:1100]  

english_test = udhr.words('English-Latin1')[0:1000] 
french_test = udhr.words('French_Francais-Latin1')[0:1000]
italian_test = udhr.words('Italian_Italiano-Latin1')[0:1000] 
spanish_test = udhr.words('Spanish_Espanol-Latin1')[0:1000]

Build unigram, bigram, and trigram character models for all four languages. You may find it convenient to use the NLTK classes FreqDist and ConditionalFreqDist, described in chapter 2 of the NLTK book (http://www.nltk.org/book/). 

For each word in the English test sets, calculate the probability assigned to that string by English vs. French unigram models, English vs. French bigram models, and English vs. French trigram models. Use the test set to report accuracy of your models. You should report the accuracies of the uni-, bi-, and tri-gram models.


Problem 2** (Language Model Comparison) (20 points). 
Perform the same experiment as above for Spanish vs. Italian. Which language pair is harder to distinguish?


Note: ITCS 5111 students will be graded out of a 100 points total for all problems in this assignment. ITCS 4111 students will be graded out of 80 points total for all problems in this assignment, except for those marked with **. 
'''
import nltk 
from nltk.corpus import udhr
from nltk import bigrams
from nltk.util import ngrams
from nltk import FreqDist
from nltk import ConditionalFreqDist


'''
	A class called "Models" is created with:
	1. a constructor -  which takes in the name of the corpus file and tokenize the first 1000 characters for unigram /bigram /trigram and finds its respective frequency distribution
	2. a method for unigram - whch taken in a word and calculate its unigram character probablity and returns that value  
	3. a method for bigram - whch taken in a word and calculate its bigram character probablity and returns that value
	4. a method for unigram - whch taken in a word and calculate its unigram character probablity and returns that value	
'''
class Models:
	
	#Constructor
	def __init__(self, corpura):

		corpus = udhr.raw(corpura)

		self.TrainingSet = corpus[0:1000]
		token = list(self.TrainingSet)

		self.Uni = token
		self.Bi = list(nltk.bigrams(token))
		self.Tri = list(nltk.trigrams(token))

		self.UniFreq = FreqDist(self.Uni)
		self.BiFreq = ConditionalFreqDist(self.Bi)
		self.TriFreq = ConditionalFreqDist(list(((w1,w2),w3) for w1,w2,w3 in self.Tri))
	
	
	#method to calculate Unigrams
	def CalUni(self, Words):
		Words = Words.strip().lower()
		Character = list(Words)

		i = 1
		for a in Character:
			i*= self.UniFreq.freq(a)

		return i


	#method to calculate Bigrams
	def CalBi(self, Words):
		Words= Words.strip().lower()
		Character=list(Words)

		i = 1
		for a,b in enumerate(Character):
			if a == 0:
				continue
			
			i*= self.BiFreq[Character[a - 1]].freq(b)

		return i


	#method to calculate Trigrams	
	def CalTri(self, Words):
		Words = Words.strip().lower()
		Character = list(Words)
		
		i=1
		for a,b in enumerate(Character):
			if a <= 1:
				continue
			i*= self.TriFreq[(Character[a - 2], Character[a - 1])].freq(b)
		
		return i


#Driver Function
def Accuracy(LangModel, Data):
	model = Models(LangModel)
	words = udhr.words(Data)[0:1000]
	WordCount = len(words)
	UniAcc = 0
	BiAcc = 0
	TriAcc = 0
	
	for word in words:
		UniP = model.CalUni(word)
		if(UniP > 0):
			UniAcc+= 1
		print("%15s - %19.18f" %(word,UniP))
	print("\t\t\t\t\t\tAtccuracy of unigram model: ", UniAcc * 100 / WordCount)
	
	for word in words:
		BiP = model.CalBi(word)
		if(BiP > 0):
			BiAcc+= 1
		print("%15s - %19.18f" %(word,BiP))
	print("\t\t\t\t\t\tAccuracy of bigram model: ", BiAcc * 100 / WordCount)
		
	for word in words:
		TriP = model.CalTri(word)
		if(TriP > 0):
			TriAcc += 1
		print("%15s - %19.18f" %(word,TriP))
	print("\t\t\t\t\t\tAccuracy of trigram model: ", TriAcc * 100 / WordCount)
	
Accuracy('English-Latin1', 'English-Latin1') 			#Call for English vs English
#Accuracy('English-Latin1', 'French_Francais-Latin1') 		#Call for English vs French
#Accuracy('Spanish_Espanol-Latin1', 'Spanish_Espanol-Latin1') 	#Call for Spanish vs Latin
#Accuracy('Spanish_Espanol-Latin1', 'Italian_Italiano-Latin1') 	#Call for Latin vs Latin



This program has been implemented on python3 on 64 bit Ubuntu 16.04 LTS version

The folder structure of the program is as follows:

For the question 1, my choice of foreign language is spanish(es).
The sentences selected for the same can be found in /data/sentences_es_dev.txt - first 10 sentences used as dev
The remaining 5 sentences in spanish can be /data/sentences_es_test.txt

Similary the corresponding source translated english can be found in:
/data/sentences_en_dev.txt - first 10 sentences translated from the source in english
/data/sentences_en_test.txt - nect 5 sentences translated from the source in english

The corresponding dictionary of the distinct words in the sentences selected using an online service in JSON can be found in:
/data/dictionary.json 
 

To run or replicate the program, use:
python3 solution1.py for the first solution - which contains the implementattion of Normal Direct Transaltion along with 6 improvements, namely:
# Improvement 1: Swap the nearest verb with the word after noun
# Improvement 2: Swap the nearest adjective with the word after noun
# Improvement 3: Bigram Language Model
# Improvement 4: Trigram Language Model
# Improvement 5: Bigram POS Language Model
# Improvement 6: Rearrangement of POS
--------------------------------------------
This will output first the Normal machine translation and then the above six improvements over Normally translated
--------------------------------------------

python3 solution2.py for the second solution -  this is the code for the solution 2  and creates translated files for both dev and test for IBM model and the improvement choice, POS tagging in this case.

The choice of dataset taken here is es-en

Bleu_score can be run on these translated files, which can be found in
as python bleu_score.py <ForeignLanguage> <TranslatedLanguage>

/es-en/dev/newstest2012.translated
/es-en/test/newstest2013.translated

/es-en/dev/newstest2012_pos.translated
/es-en/test/newstest2013_pos.translated
--------------------------------------------------------------------
Report file contains the solution to the third part of the question.
--------------------------------------------------------------------
 

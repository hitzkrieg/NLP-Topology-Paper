import codecs
import numpy as np
import sys
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

print ('Number of arguments:', len(sys.argv), 'arguments.')
print ('Argument List:', str(sys.argv))

if len(sys.argv)!=3 :
	print("Arguments wrongly provided. Use -story or -poem as arguments.")
	sys.exit(0)


str=""

f = codecs.open(sys.argv[1], encoding='utf-8')
for line in f:
      str= str + line

str_list= str.splitlines()
str_list= [word for word in filter(None, str_list)]
#print(str_list)

if sys.argv[2] == '-story':
	vectorizer = CountVectorizer(min_df=1, stop_words= 'english', token_pattern=r"\b\w+\b")
elif sys.argv[2] == '-poem':
	vectorizer = CountVectorizer(min_df=1, token_pattern=r"\b\w+\b")
else:
	print("Wrong option given as argument. Use -story or -poem as arguments.")
	sys.exit(0)


#To view the way the tokenization will happen
#analyze = vectorizer.build_analyzer()
#print (analyze(str_list[1]))

tf = vectorizer.fit_transform(str_list)
ans= tf.toarray()
print(ans.shape)




my_df = pd.DataFrame(ans)
my_df.to_csv('output.text', sep= ' ', index=False, header=False)



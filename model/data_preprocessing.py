"""
Data used format 

Sentiment	SentimentText
0	is so sad for my APL friend.............
0	I missed the New Moon trailer...
1	omg its already 7:30 :O
0	.. Omgaga. Im sooo  im gunna CRy. I've been at this dentist since 11.. I was suposed 2 just get a crown put on (30mins)...
0	i think mi bf is cheating on me!!!       T_T
0	or i just worry too much?        
1	Juuuuuuuuuuuuuuuuussssst Chillin!!
0	Sunny Again        Work Tomorrow  :-|       TV Tonight
1	handed in my uniform today . i miss you already

converted using this file
"""

import csv
reader = csv.reader(open('Sentiment_Analysis_Dataset.csv', 'r'),skipinitialspace=True)
writer = csv.writer(open('data.csv', 'w'),delimiter="\t")
for row in reader:
    
    writer.writerow([row[1] ,row[3]])
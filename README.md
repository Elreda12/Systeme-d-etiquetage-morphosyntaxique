# Parts-of-Speech Tagging (POS)

 On realise un syst`eme d’´etiquetage morphosyntaxique pour la langue
fran¸caise en utilisant des donn´ees annot´ees de votre choix ou les donn´ees indiqu´ees dans le lien suivant:
https://github.com/qanastek/ANTILLES/tree/main/ANTILLES

# Part 1: Parts-of-speech tagging

# Part 1.1 - Training
You will start with the simplest possible parts-of-speech tagger and we will build up to the state of the art.

In this section, you will find the words that are not ambiguous.

For example, the word is is a verb and it is not ambiguous.
In the WSJ corpus,  86
 % of the token are unambiguous (meaning they have only one tag)
About  14%
  are ambiguous (meaning that they have more than one tag)
  ![dd](https://github.com/Elreda12/un-systeme-d-etiquetage-morphosyntaxique/assets/141457934/4cd5af7c-634b-433d-a644-8ac82d51e06e)
  
  Before you start predicting the tags of each word, you will need to compute a few dictionaries that will help you to generate the tables.

  # Transition counts
The first dictionary is the transition_counts dictionary which computes the number of times each tag happened next to another tag.
This dictionary will be used to compute:
𝑃(𝑡𝑖|𝑡𝑖−1)
This is the probability of a tag at position  𝑖
  given the tag at position  𝑖−1
 .

In order for you to compute equation 1, you will create a transition_counts dictionary where

The keys are (prev_tag, tag)
The values are the number of times those two tags appeared in that order.

# Emission counts
The second dictionary you will compute is the emission_counts dictionary. This dictionary will be used to compute:

𝑃(𝑤𝑖|𝑡𝑖)
In other words, you will use it to compute the probability of a word given its tag.

In order for you to compute equation 2, you will create an emission_counts dictionary where

The keys are (tag, word)
The values are the number of times that pair showed up in your training set.

# Tag counts
The last dictionary you will compute is the tag_counts dictionary.

The key is the tag
The value is the number of times each tag appeared.

 # Part 1.2 - Testing
Now you will test the accuracy of your parts-of-speech tagger using your emission_counts dictionary.

Given your preprocessed test corpus prep, you will assign a parts-of-speech tag to every word in that corpus.
Using the original tagged test corpus y, you will then compute what percent of the tags you got correct.

# Part 3: Viterbi Algorithm and Dynamic Programming
You will use your two matrices, A and B to compute the Viterbi algorithm. We have decomposed this process into three main steps for you.

Initialization - In this part you initialize the best_paths and best_probabilities matrices that you will be populating in feed_forward.
Feed forward - At each step, you calculate the probability of each path happening and the best paths up to that point.
Feed backward: This allows you to find the best path with the highest probabilities.

# Part 3.1: Initialization
You will start by initializing two matrices of the same dimension.

best_probs: Each cell contains the probability of going from one POS tag to a word in the corpus.

best_paths: A matrix that helps you trace through the best possible path in the corpus.


Both matrices will be initialized to zero except for column zero of best_probs.

Column zero of best_probs is initialized with the assumption that the first word of the corpus was preceded by a start token ("--s--").
This allows you to reference the A matrix for the transition probability
Here is how to initialize column 0 of best_probs:

The probability of the best path going from the start index to a given POS tag indexed by integer 𝑖
 is denoted by best_probs[𝑠𝑖𝑑𝑥,𝑖]
.
This is estimated as the probability that the start tag transitions to the POS denoted by index 𝑖
: 𝐀[𝑠𝑖𝑑𝑥,𝑖]
 AND that the POS tag denoted by 𝑖
 emits the first word of the given corpus, which is 𝐁[𝑖,𝑣𝑜𝑐𝑎𝑏[𝑐𝑜𝑟𝑝𝑢𝑠[0]]]
.
Note that vocab[corpus[0]] refers to the first word of the corpus (the word at position 0 of the corpus).
vocab is a dictionary that returns the unique integer that refers to that particular word.
Conceptually, it looks like this: best_probs[𝑠𝑖𝑑𝑥,𝑖]=𝐀[𝑠𝑖𝑑𝑥,𝑖]×𝐁[𝑖,𝑐𝑜𝑟𝑝𝑢𝑠[0]]
In order to avoid multiplying and storing small values on the computer, we'll take the log of the product, which becomes the sum of two logs:

𝑏𝑒𝑠𝑡_𝑝𝑟𝑜𝑏𝑠[𝑖,0]=𝑙𝑜𝑔(𝐴[𝑠𝑖𝑑𝑥,𝑖])+𝑙𝑜𝑔(𝐵[𝑖,𝑣𝑜𝑐𝑎𝑏[𝑐𝑜𝑟𝑝𝑢𝑠[0]]
Also, to avoid taking the log of 0 (which is defined as negative infinity), the code itself will just set 𝑏𝑒𝑠𝑡_𝑝𝑟𝑜𝑏𝑠[𝑖,0]=𝑓𝑙𝑜𝑎𝑡(′−𝑖𝑛𝑓′)
 when 𝐴[𝑠𝑖𝑑𝑥,𝑖]==0
So the implementation to initialize 𝑏𝑒𝑠𝑡_𝑝𝑟𝑜𝑏𝑠
 looks like this:

𝑖𝑓𝐴[𝑠𝑖𝑑𝑥,𝑖]<>0:𝑏𝑒𝑠𝑡_𝑝𝑟𝑜𝑏𝑠[𝑖,0]=𝑙𝑜𝑔(𝐴[𝑠𝑖𝑑𝑥,𝑖])+𝑙𝑜𝑔(𝐵[𝑖,𝑣𝑜𝑐𝑎𝑏[𝑐𝑜𝑟𝑝𝑢𝑠[0]]])
𝑖𝑓𝐴[𝑠𝑖𝑑𝑥,𝑖]==0:𝑏𝑒𝑠𝑡_𝑝𝑟𝑜𝑏𝑠[𝑖,0]=𝑓𝑙𝑜𝑎𝑡(′−𝑖𝑛𝑓′)

# Part 3.2 Viterbi Forward
You will implement the viterbi_forward segment. In other words, you will populate your best_probs and best_paths matrices.

Walk forward through the corpus.
For each word, compute a probability for each possible tag.
Unlike the previous algorithm predict_pos (the 'warm-up' exercise), this will include the path up to that (word,tag) combination.
Here is an example with a three-word corpus "Loss tracks upward":

Note, in this example, only a subset of states (POS tags) are shown in the diagram below, for easier reading.
In the diagram below, the first word "Loss" is already initialized.
The algorithm will compute a probability for each of the potential tags in the second and future words.
Compute the probability that the tag of the second work ('tracks') is a verb, 3rd person singular present (VBZ).

In the best_probs matrix, go to the column of the second word ('tracks'), and row 40 (VBZ), this cell is highlighted in light orange in the diagram below.
Examine each of the paths from the tags of the first word ('Loss') and choose the most likely path.
An example of the calculation for one of those paths is the path from ('Loss', NN) to ('tracks', VBZ).
The log of the probability of the path up to and including the first word 'Loss' having POS tag NN is  −14.32
 . The best_probs matrix contains this value -14.32 in the column for 'Loss' and row for 'NN'.
Find the probability that NN transitions to VBZ. To find this probability, go to the A transition matrix, and go to the row for 'NN' and the column for 'VBZ'. The value is  4.37𝑒−02
 , which is circled in the diagram, so add  −14.32+𝑙𝑜𝑔(4.37𝑒−02)
 .
Find the log of the probability that the tag VBS would 'emit' the word 'tracks'. To find this, look at the 'B' emission matrix in row 'VBZ' and the column for the word 'tracks'. The value  4.61𝑒−04
  is circled in the diagram below. So add  −14.32+𝑙𝑜𝑔(4.37𝑒−02)+𝑙𝑜𝑔(4.61𝑒−04)
 .
The sum of  −14.32+𝑙𝑜𝑔(4.37𝑒−02)+𝑙𝑜𝑔(4.61𝑒−04)
  is  −25.13
 . Store  −25.13
  in the best_probs matrix at row 'VBZ' and column 'tracks' (as seen in the cell that is highlighted in light orange in the diagram).
All other paths in best_probs are calculated. Notice that  −25.13
  is greater than all of the other values in column 'tracks' of matrix best_probs, and so the most likely path to 'VBZ' is from 'NN'. 'NN' is in row 20 of the best_probs matrix, so  20
  is the most likely path.
Store the most likely path  20
  in the best_paths table. This is highlighted in light orange in the diagram below.
The formula to compute the probability and path for the 𝑖𝑡ℎ
 word in the 𝑐𝑜𝑟𝑝𝑢𝑠
, the prior word 𝑖−1
 in the corpus, current POS tag 𝑗
, and previous POS tag 𝑘
 is:

prob=𝐛𝐞𝐬𝐭_𝐩𝐫𝐨𝐛𝑘,𝑖−1+log(𝐀𝑘,𝑗)+log(𝐁𝑗,𝑣𝑜𝑐𝑎𝑏(𝑐𝑜𝑟𝑝𝑢𝑠𝑖))
where 𝑐𝑜𝑟𝑝𝑢𝑠𝑖
 is the word in the corpus at index 𝑖
, and 𝑣𝑜𝑐𝑎𝑏
 is the dictionary that gets the unique integer that represents a given word.

path=𝑘
where 𝑘
 is the integer representing the previous POS tag.

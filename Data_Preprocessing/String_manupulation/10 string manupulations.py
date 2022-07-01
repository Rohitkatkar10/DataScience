# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 09:43:12 2022

@author: rohit
"""

### string manipulations ################

#  Que 1.Create a string “Grow Gratitude”.
string = "Grow Gratitide"

# access the 'G' of 'grow'
print(string.index('G')) # this will return index of first occurence of "G".
# OR
first_G = string.find("G") 
print(first_G )

print(string[0])

# length of the string
print(len(string))

# count "G" in string
words = string.split(' ')
count = 0
for word in words:
    for letter in word:
        if letter == "G":
            count = count + 1
print(count) # count = 2
            
# Que 2.Create a string “Being aware of a single shortcoming within yourself is far more useful
#        than being aware of a thousand in someone else.”

string2 = "Being aware of a single shortcoming within yourself is far more useful than being aware of a thousand in someone else."            
# number of character in string2
print(len(string2))     # 118        


# Que 3. Create a string "Idealistic as it may sound, altruism should be the driving force in business, 
#        not just competition and a desire for wealth"
 
string3 = "Idealistic as it may sound, altruism should be the driving force in business, not just competition and a desire for wealth"
  
# one char of the word
print(string3[14])          
            
# get first three chara
print(string3[0:3])

# Get last three chara
print(string3[-3:]) # -1 = h, -2 = t, -3 = l, .....

# Que 4. create a string "stay positive and optimistic". 
#        Now write a code to split on whitespace.

string4 = "stay positive and optimistic"

# split on whitespace
words = string4.split(" ") # give space as delimeter
print(words)

# The string starts with “s”
if string4.startswith("s"):
    print(string4)

# the string starts with "H"
if string4.startswith("H"):
    print(string4)
    
    
# the string starts with "d"
if string4.startswith("d"):
    print(string4)

# the string starts with "c"
if string4.startswith("c"):
    print(string4)
    
    
# 5.Write a code to print " 🪐 " one hundred and eight times    

limit = 108
x = " 🪐 " *108
print(x)

# 7.Create a string “Grow Gratitude” and write a code to replace “Grow” with “Growth of”
string7 = 'Grow Gratitude'

print(string7.replace('Grow','Growth of')) # Growth of Gratitude
print(string7) # Grow Gratitude

New_string = string7.replace('Grow','Growth of') # for permanent change
print(New_string)


# Make story correct
story = ".elgnujehtotniffo deps mehtfohtoB .eerfnoilehttesotseporeht no dewangdnanar eh ,ylkciuQ  \
.elbuortninoilehtdecitondnatsapdeklawesuomeht ,nooS .repmihwotdetratsdnatuotegotgnilggurts saw noilehT \
.eert a tsniagapumihdeityehT .mehthtiwnoilehtkootdnatserofehtotniemacsretnuhwef a ,yad enO \
.ogmihteldnaecnedifnocs’esuomeht ta dehgualnoilehT ”.emevasuoy fi yademosuoyotplehtaergfo eb lliw I ,uoyesimorp I“ \
.eerfmihtesotnoilehtdetseuqeryletarepsedesuomehtnehwesuomehttaeottuoba saw eH \
.yrgnaetiuqpuekow eh dna ,peels s’noilehtdebrutsidsihT .nufroftsujydobsihnwoddnapugninnurdetratsesuom \
a nehwelgnujehtnignipeelsecno saw noil A"

# \ used to make string  one if string goes on next line. 
print(story.reverse) # not working 

# use join function
word_story = story.split(" ") # split story into words

correct_story = ' '.join(word[::-1] for word in word_story) # correct every word 

correct_story= correct_story.split(" ") # again split corrected word in list format
# check type of variable
type(correct_story)  # its list

new_story = correct_story[::-1] # then reveres the corrected word list

new_story = ' '.join(word for word in new_story) # join the list of corrected word to get correct story sequence.

print(new_story) 


######## End of script #######
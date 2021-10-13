import numpy as np
from collections import defaultdict
from graphviz import Digraph
import pandas as pd
import random
from copy import deepcopy

class Node():
    '''
    Node class, for root and internal nodes. Leaves are values pure
    '''
    def __init__(self, attr, values):
        super().__init__()
        
        self.attribute = attr
        self.values = values
        self.children = []
    
    def get_attribute(self):
        return self.attribute

def plurality_value(parent_example):
    '''
    parent_example : dataframe, of examples
    The value that is most dominant for this value
    '''
    check =  parent_example["Survived"].value_counts()

    if check[0] > check[1]: 
        return 0
    elif check[0] < check[1]:
        return 1
    else:
        return random.randint(0,1)

def equal_classification_check(list_of_examples):
    '''
    Check if all classification is equal for all examples
    '''
    #counts the classification
    check = list_of_examples["Survived"].value_counts()
    if len(check) < 2:
        return True
    return False

def equal_classification_value(examples):
    '''
    returns value when classification is equal for all examples
    '''
    #counts the classification
    check = examples["Survived"].value_counts()
    #returns the classification
    return int(check.index[0])

    
def information_gain(attr, examples, attribute_value_pairs, continous = 0):
    '''
    return value of information gain,
    equals Entropy - remainder = H - Remainer
    Tried to implement the part for the continous variable as well, but did not make it before deadline.. Tried to write in pseudo what i would do further
    '''

    def B(p, n):
        '''
        returns entropy with given values
        '''
        q = p/(p+n)
        return -(q*np.log2(q) + (1-q)*np.log2(1-q))

    split = 0   #used for rememering where the split were in continous variables. 
    positive = examples["Survived"].value_counts()[1]   #positive values, amount of survived persons
    negative = examples["Survived"].value_counts()[0]   #negative values, amount of not survived
    remainder = 0   #keeping track of remainding entropy

    if (continous): #supposed to use when continous variables
        #The point were to add to a list all thresholds with entropy it gives. 
        #Tar dette på norsk. Tanken var å legge til grenseverdi, attributt og entropi som en tuppel i en liste. I DTL ville det da ha blitt lagt til den grenseverdien som ga lavest entropy, og denne ville blitt lagt til i lista som best-variabelen velger fra. Den ville da etter at split hadde blitt funnet her oppført seg som en vanlig klassifiserbar variabel og verdier ved test hadde sjekket om de var høyere eller lavere enn denne.
        for temp_threshold in attribute_value_pairs.get(attr):  
            new_frame = examples[examples[attr] <= temp_threshold]['Survived'].value_counts()
            if len(new_frame) < 2:
                continue
            else:
                negative_with_value = new_frame[0]
                positive_with_value = new_frame[1]
                remainder += ((positive_with_value + negative_with_value)/(positive + negative))*B(positive_with_value, negative_with_value)
                split = 0
    else: #Her er koden for uten continous.
        for value in attribute_value_pairs.get(attr):
            new_frame = examples[examples[attr] == value]['Survived'].value_counts() #filtrerer examples til ny frame med verdien satt.
            if len(new_frame) < 2:  #klassifiseringsproblem, ikkeeksisterende entropi.
                continue
            else:
                negative_with_value = new_frame[0] #ny verdi for ikke overlevende med gitt verdi
                positive_with_value = new_frame[1]  #ny verdi for overlevende med gitt verdi
                remainder += ((positive_with_value + negative_with_value)/(positive + negative))*B(positive_with_value, negative_with_value)
                    #beregner remainder for å finne reell entropi med gitt valg
    return (attr, B(positive, negative) - remainder, split) #split skulle retunrere splittverdi, brukes ikke i classification

def tree_construct(attr, att_value_list):
    '''
    Trekonstruktør, lager node med attributt og verdier satt.
    '''
    tree_node = Node(attr, att_value_list.get(attr))
    return tree_node


def decision_tree_learning(examples, attributes, parent_examples):
    '''
    Decision tree construct, after figure 18.5 in AIMA
    '''
    attribute_value_pairs = {}
    for att in attributes:
        attribute_value_pairs[att] = examples[att].unique() #Legger til attributter og dens unike verdier
    
    if len(examples) == 0:  #hvis ingen eksempler, sjekk dominant til foreldere
        return plurality_value(parent_examples)

    elif equal_classification_check(examples): #hvis alle eksempler gir ut lik verdi, retunrer denne verdien
        return equal_classification_value(examples)

    elif len(attributes) == 0 or (len(attributes) == 1 and attributes[0] == "Survived"): #hvis attribut er tom, retunrer dominant for eksempler
        return plurality_value(examples)

    else: #creating a subtree now
        temp_att = [] #brukes for classification noder
        cont_att = [] #mellomlagring for continous, her 3-tuple blir lagret fra linje 97. Velger så ut høyest infoGain fra denne inn i temp_att.
        for att in attributes:
            if att == 'Survived':
                continue #har med survived i attributter, men skal ikke sjekke på denne.
            if att == 'Parch' or att == 'SibSp':    #Her continous skulle ha blitt sjekket opp med threshold og lagt inn in cont_att. Er hardkodet her, ville gjort det mer fleksibelt med en sjekk istedenfor hardkodete variabler. 
                print(attribute_value_pairs.get(att))
                cont_att.append(information_gain(att, examples, attribute_value_pairs, 1))
                pass
            else: #hit den går ved classification. legger til alle attributter med tilhørende infoGain
                temp_att.append(information_gain(att, examples, attribute_value_pairs))
        
        #correct_split = max(cont_att, key=lambda item:item[1])[2] #uthenting av beste verdi for splitt. attributt, infoGain og splitt.
        #temp_att.append(correct_split) #legge til verdi for continous attribute med høyest information gain, correct split
        best = max(temp_att, key=lambda item:item[1])[0] #velger ut variabel som har høyest infoGAin.
        
        if best ==  'Parch' or best == 'SibSp': #her man skal hente ut splittverdien. element 3 i tuppelen.
            split = max(temp_att, key=lambda item:item[1])[2]

        tree = tree_construct(best, attribute_value_pairs) #konstruerer tre (noden) til best
        for value in tree.values: #itererer gjennom verdiene til tre
            new_examples = examples[examples[tree.attribute]==value] #filtrerer til riktig eksempelsett med satt value på attributt
            new_attributes = deepcopy(attributes)   #kopierer slik at man ikke endrer på noe man ikke burde
            #print("new att: ", new_attributes, "tree.attribute: ", tree.attribute)
            new_attributes.remove(tree.get_attribute())     #fjerner attributt fra lista
            subtree = decision_tree_learning(new_examples, new_attributes, examples) #rekursivt dypere i treet med ny eksempel, attributt og tilpasset verdi
            tree.children.append((subtree, value)) #leggger til i treet, blir barn/subtre
        return tree #returnerer resultat

def predict(tree, sample):
    '''
    Lucky here that none of the values in example set are of equal values... Must change for continous variables when it is several values of 0/1 etc.
    Works on categorical though and with the 2 (or three with embarked) in a), but no equal values
    '''
    examples = sample.iloc[:,1:].values #data to predict from
    solution = sample.iloc[:,0].values  #solution for data
    true_tree = deepcopy(tree)          #copy tree, do not want to change something not supposed to
    score = 0
    total_amount = len(solution)
    count=0
    #print(tree)
    #print(tree.children)
    # TODO edit to handle continous values. Afterall, this works only when no values in examples are equal. 1a is good, but if several 0s or 1s as representation for True/false, this will fail. Not a optimal loop, but works for a) :) Må endre denne til å fungere for continous
    def pred(testdata, tree):
        while not(isinstance(tree, int)):   #så lenge det ikke er løvnode
            for value in tree.children: #for hver verdi til noden
                for i in testdata:      #for hver verdi i data å predikere fra
                    if i == value[1]:   #hvis verdien er lik verdien til children til attributten (Ja, denne som er helt høl i hue men funka på a)
                        tree = value[0] #setter treet til barnnoden, går dypere
                        break
                    continue
                continue
            continue
        if tree == solution[count]: #hvis riktig svar, return 1
            return 1
        return 0 
    
    for test in examples:
        score += pred(test, tree)   #sjekker om rikitg
        tree = true_tree    #setter treet til rotnoden igjen
        count+=1    #oppdaterer count, følger antall eksempelr
        
    return score/total_amount   #retunrer resultat, presisjon

def task_1a():
    '''
    Drew the trees manually from DOT-files
    Task 1a, dataset filtered to only contain categorical variables. Continous and variables with missing values removed.
    '''
    attributes = ["Survived", "Pclass", "Sex"] #Fjern, "Embarked"
    df = pd.read_csv("C:/prog/TDT4171/titanic/train.csv", usecols=attributes)

    df_test = pd.read_csv("C:/prog/TDT4171/titanic/test.csv", usecols=attributes)

    new_tree = decision_tree_learning(df, attributes, df)
    #draw_tree(new_tree)
    result = predict(new_tree, df_test)
    print("Accuracy of decision network: ",result)

task_1a()    

def task_1b():
    attributes = ["Survived", "Pclass", "Sex", "SibSp", "Parch"] #Fjern ,, "Embarked" 
    df = pd.read_csv("C:/prog/TDT4171/titanic/train.csv", usecols=attributes)
    new_tree = decision_tree_learning(df, attributes, df)
    #draw_tree(new_tree)
    #result = predict(new_tree, df_test)
    #print("Accuracy of decision network: ",result)

#task_1b()
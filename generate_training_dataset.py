import glob
import re
import numpy as np
from stanford_parser_wrapper import Parser
import cPickle

def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\(.*?\)", "", string)
    string = re.sub(r"\s{2,}", " ", string)       
    return string.strip()

def build_datasets_sick():
    parser = Parser() 
    folders = ['train', 'dev', 'test']
    
    for folder in folders:
        index = 0
        dataset = []
        a_s = "./sick/"+folder+"/a.txt"
        b_s = "./sick/"+folder+"/b.txt"
        sims = "./sick/"+folder+"/sim.txt"
        
        with open(a_s, "rb") as f1, open(b_s, "rb") as f2, open(sims, "rb") as f3:                            
            for a, b, sim in zip(f1,f2,f3):
                index += 1
                if index % 200 == 0:
                    print index
                
                first_sent = clean_str(a)
                second_sent = clean_str(b)
                
                if len(first_sent) ==0 or len(second_sent) ==0:
                    continue
                if " " not in first_sent or " " not in second_sent:
                    continue

                try:
                    first_parse_output = parser.parseSentence(first_sent)
                except:
                    print "first_sentence can't be parsing"
                    #print first_sentence
                    #traceback.print_exc()
                    continue
                try:
                    second_parse_output = parser.parseSentence(second_sent)
                except:
                    print "second_sentence can't be parsing"
                    #print second_sentence
                    #traceback.print_exc()
                    continue
    
                datum = {   "score":sim.strip(), 
                            "text": (first_sent, second_sent), 
                            "parse":(first_parse_output, second_parse_output)
                        }
                dataset.append(datum)            

        with open(folder+"_dataset","wb") as f:
            cPickle.dump(dataset,f)

if __name__ == "__main__":
    print("=" * 80)
    print "Preprocessing SICK dataset"
    print("=" * 80)











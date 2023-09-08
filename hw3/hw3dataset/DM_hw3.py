import os
import sys
import DM_src


if __name__ == "__main__":
    if str(sys.argv[1]) == "graph_1":
        with open('hw3dataset/graph_1.txt','r') as f:
            lines = f.readlines()
            for line in lines:
                x = line.split()
                for i in x:
                    y = i.split(',')
                    print(y,type(y)) #list
                    #for num in y:


                
                #y = int(x.split(','))
    # elif str(sys.argv[1]) == "graph_2":
    #     with open('graph_2.txt','r') as f: 


    # elif str(sys.argv[1]) == "graph_3":
    #     with open('graph_3.txt','r') as f:

    # elif str(sys.argv[1]) == "graph_4":
    #     with open('graph_4.txt','r') as f:

    # elif str(sys.argv[1]) == "graph_5":
    #     with open('graph_5.txt','r') as f:

    # elif str(sys.argv[1]) == "graph_6":
    #     with open('graph_6.txt','r') as f:


    
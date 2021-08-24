import sys, getopt
import re


food = []
runs = 1
filename = 'Run_1'

def parser(argv):

    global food, runs, filename

    try:
        opts, args = getopt.getopt(argv, 'f:r:n', ["food=", "runs=", "fn="])
    except getopt.GetoptError:
        print('Arguments do not match')
        sys.exit(2)
    
    for opt, arg in opts:
        
        if opt in ('-f', "--food"):
            if re.match('[()]', arg):
                food = list(arg)
            else:
                food = arg
        
        if opt == '-r' or opt == '--runs':
            runs = arg

        if opt == '-n' or opt == '--fn':
            filename = arg



    
    
if __name__ == '__main__':
    parser(sys.argv[1:])
    print(food)
    print(type(food))
    print(runs)
    print(filename)
    


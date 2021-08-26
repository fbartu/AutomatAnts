import sys, getopt
import re
import params


def argparse(argv):

    if 'params' not in globals():
        import params

    error_msg_rec = 'Recruitment misspecified ! Check possible options in help or params.py'
    error_msg_float = 'Parameter must be a float number !'
    error_msg_int = 'Parameter must be an integer !'
    error_msg_bool = 'Parameter must be a boolean !'
    wrng_msg_1 = 'Leaving default parameter value ...'
    wrng_msg_2 = 'Exiting program ...'

    
    try:
        opts, args = getopt.getopt(argv, 'p:f:r:n:m:h',
         ["params=", "food=", "recruitment=", "nruns=", "mvfood=","help="])
    except getopt.GetoptError:
        print(wrng_msg_2)
        sys.exit(2)

    if '--foodamount' in opts:
        try:
            globals()['params'].foodXvertex = int(args[opts.index('--foodamount')])
        except:
            print(error_msg_int)
            print(wrng_msg_1)

    for opt, arg in opts:

        if opt in ('-p', '--params'):
            paramlist = arg.split(',')
            for param in paramlist:
                param = param.split('=')
                if hasattr(globals()['params'], param[0]):
                    try:
                        setattr(globals()['params'], param[0], float(param[1]))
                    except:
                        print(error_msg_float)
                        print(wrng_msg_1)
                
                else:
                    print('Parameter ' + str(param[0]) + ' not found in params.py')
                    print(wrng_msg_1)
        
        if opt in ('-f', '--food'):
            # find by tuples
            pos = re.findall(r'\((.*?,.*?)\)', arg)
            pos = [(int(i[0]), int(i[2])) for i in pos]
            globals()['params'].food = dict.fromkeys(pos, globals()['params'].foodXvertex)

        if opt in ('-r', '--recruitment'):
            if arg in ('NR', 'IR', 'HR', 'GR'):
                globals()['params'].recruitment = arg

            else:
                print(error_msg_rec)
                print(wrng_msg_1)

        if opt in ('-n', '--nruns'):

            l = arg.split(',')
            try:
                globals()['params'].n_runs = int(l[0])
            
            except:
                print(error_msg_int)
                print(wrng_msg_1)

            if len(l) == 2:

                if l[1] == 'True':
                    globals()['params'].run_parallel = True

                elif l[1] == 'False':
                    globals()['params'].run_parallel = False

                else:
                    print(error_msg_bool)
                    print(wrng_msg_1)

        if opt in ('-m', '--mvfood'):
            l = arg.split(',')
            try:
                l = (int[0], int[1])
                if sum(l) % 2 == 0:
                    fk = globals()['params'].food.keys()
                    food = [(i[0] + l[0], i[1] + l[1]) for i in list(fk)]
                    globals()['params'].food = dict.fromkeys(food, globals()['params'].foodXvertex)
                else:
                    print('Sum of coordinates must be even !')
                    print(wrng_msg_1)
                    
            except:
                print('Parameter must be two integers, whose sum must be even !')
                print(wrng_msg_1)


        if opt in ('-h', '--help'):
            '''PRINT HELP...'''
            pass


    
if __name__ == '__main__':
    
    # import params

    print('Food: ')
    print(params.food)

    print('--------------------- \n')
    print('Parameters: ')
    print("alpha  = " + str(params.alpha))
    print("beta_1 = " + str(params.beta_1))
    print("beta_2 = " + str(params.beta_2))
    print("gamma_1 = " + str(params.gamma_1))
    print("gamma_2 = " + str(params.gamma_2))
    print("omega = " + str(params.omega))
    print("eta = " + str(params.eta))
    print("mu = " + str(params.mu))
    print('--------------------- \n')

    print('Parallel running... ?')
    print(params.n_runs)
    print(params.run_parallel)


    argparse(sys.argv[1:])
    print('Food new: ')
    print(params.food)

    print('--------------------- \n')
    print('Parameters new: ')
    print("alpha  = " + str(params.alpha))
    print("beta_1 = " + str(params.beta_1))
    print("beta_2 = " + str(params.beta_2))
    print("gamma_1 = " + str(params.gamma_1))
    print("gamma_2 = " + str(params.gamma_2))
    print("omega = " + str(params.omega))
    print("eta = " + str(params.eta))
    print("mu = " + str(params.mu))
    print('--------------------- \n')

    print('Parallel running... ?')
    print(params.n_runs)
    print(params.run_parallel)
    


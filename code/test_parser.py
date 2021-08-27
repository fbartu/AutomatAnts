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
         ["params=", "food=", "recruitment=", "nruns=", "mvfood=","help=", "foodamount="])
    except getopt.GetoptError:
        print('Something went wrong ! Try typing -h or --help to see possible parameters.')
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
            pos = re.findall(r'\[(.*?,.*?)\]', arg)
            pos = [(int(i[0]), int(i[2])) for i in pos]
            globals()['params'].food = dict.fromkeys(pos, globals()['params'].foodXvertex)

        if opt in ('-r', '--recruitment'):

            arg = arg.split(',')
            if len(arg) == 1:
                globals()['params'].recruitment = (''.join(arg), '')
            
            elif len(arg) == 2:
                globals()['params'].recruitment = (''.join(arg[0]), ''.join(arg[1]))

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

            print('HELP HAS BEEN SUMMONED !!')

            if arg == 'summary':
                print('The possible options are')
                print('\n')
                print('-p or --param to change individual or multiple cellular automata parameters')
                print('-f or --food to specify food node positions (as list of coordinates)')
                print('-r or --recruitment to specify recruitment strategy')
                print('-n or --nruns to set the number of runs (replicas) and whether they should be run in parallel')
                print('-m or --mvfood to move food position, relative to the default position')

            if arg.split('_')[0] == 'e':
                arg = arg.split('_')
                for a in arg[1:]:
                    if a in ('p', 'param'):
                        print('-p or --param for parameter change:')
                        print('\t list of parameters that must be typed like parameter1=value,parameter2=value,parameter3=value')
                        print('\t EXAMPLE: eta=0.05,gamma_1=1.2 ; <- this will set parameters eta to 0.05 and gamma_1 to 1.2')
                        print('\t EXAMPLE: mu=1,gamma_1=0.5,gamma_2=1.5,alpha=0 ; <- this will set parameters mu to 1.0, gamma_1 to 0.5, gamma_2 to 1.5 and alpha to 0.0')
                        print('\t if any paramater is misspecified or ignored, parameter value will be left by default.')
                        print('\t notice each pair parameter value is separated by = symbol, whilst pairs are separated by commas. Avoid using spaces !')
                    
                    elif a in ('f', 'food'):
                        pass

                    elif a in ('r', 'recruitment'):
                        pass

                    elif a in ('n', 'nruns'):
                        pass

                    elif a in ('m', 'mvfood'):
                        pass

                    else:
                        print('Specified parameter "' + str(a) + '" not found !')


            else:

                print('For a shorter version, it is possible to call -h summary')
                print('\n')
                print('For examples, you can type -h e_@parameter (i.e -h e_param; -h e_recruitment')
                print('Both short and long options can be used (alternatives would be -h e_p; -h e_r)')
                print('Multiple examples can be displayed at once (i.e -h e_p_r')
                
                '''
                -h e_param = example param
                -h summary = display only a one line description of available parameters
                '''

                print('The list of possible parameters to change are:')
                print('\n')
                print('-p or --param for parameter change:')
                print('\t list of parameters that must be typed like parameter1=value,parameter2=value,parameter3=value')
                print('\t if any paramater is misspecified or ignored, parameter value will be left by default.')
                print('\t notice each pair parameter value is separated by = symbol, whilst pairs are separated by commas. Avoid using spaces !')
                print('\n')
                print('---------------------------------------')
                print('\n')
                print('-f or --food for specifying food positions:')
                print('\t list of food nodes must be typed like [x1,y1],[x2,y2],[x3,y3]')
                print('\t example')
                print('\n')
                print('---------------------------------------')
                print('\n')
                print('-r or --recruitment to specify recruitment type')
                print('\t list of two elements; the recruitment strategy and whether recruitment is parallel or not')
                print('\t the options are: ')
                print('\t parameter 1:')
                print('\t \t NR: No recruitment')
                print('\t \t IR: Individual recruitment (a single ant is recruited)')
                print('\t \t GR: Group recruitment (3 to 5 ants are recruited)')
                print('\t \t HR: Hybrid recruitment (0 to 5 ants are recruited)')
                print('\t parameter 2:')
                print('\t \t s: Serial recruitment (explotation of food patches does not occur simultaneously)')
                print('\t \t else: Any other value will lead to default behaviour')
                print('\n')
                print('---------------------------------------')
                print('\n')
                print('-n or --nruns to specify number of runs (replicas) and whether or not to run in parallel')
                print('\t list of the kind integer,boolean ; such as 100,True (100 runs in parallel)')
                print('\n')
                print('---------------------------------------')
                print('\n')
                print('-m or --mvfood to move the foodpatches across the grid')
                print('\t a pair of integer values: x,y; whose sum must be even, to displace patches in number of hexagons')
                print('\n')
                print('---------------------------------------')
                print('\n')
                print('--foodamount to set the number of foodpieces per node')
                print('\t an integer is expected') 

            print('\n')
            print('---------------------------------------')
            print('\n')    
            print('\n')
            print('HELP ENDS HERE ... Exiting program.')
            sys.exit(0)


    
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
    


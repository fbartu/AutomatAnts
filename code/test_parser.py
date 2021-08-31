# import getopt # debugging
from argparser import argparse
import sys
import params

if __name__ == '__main__':


    # print('Food: ')
    # print(params.food)

    # print('--------------------- \n')
    # print('Parameters: ')
    # print("alpha  = " + str(params.alpha))
    # print("beta_1 = " + str(params.beta_1))
    # print("beta_2 = " + str(params.beta_2))
    # print("gamma_1 = " + str(params.gamma_1))
    # print("gamma_2 = " + str(params.gamma_2))
    # print("omega = " + str(params.omega))
    # print("eta = " + str(params.eta))
    # print("mu = " + str(params.mu))
    # print('--------------------- \n')

    # print('Parallel running... ?')
    # print(params.n_runs)
    # print(params.run_parallel)

    # print('--------------------- \n')

    # print('Recruitment')
    # print(params.recruitment)
    # print('--------------------- \n')
    # print('Food amount')
    # print(params.foodXvertex)
    # print('--------------------- \n')
    # print('File name')
    # print(params.file_name)
    # print('Folder name')
    # if 'folder' in globals():
    #     print(globals()['folder'])
    # else:
    #     print('Folder name does not exist')

    # print('--------------------- \n')

    argparse(sys.argv[1:])

    print('\n')
    print('\n')
    print('--------------------- \n')
    

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

    print('--------------------- \n')

    print('Recruitment')
    print(params.recruitment)
    print('--------------------- \n')
    print('Food amount')
    print(params.foodXvertex)
    print('--------------------- \n')
    print('File name')
    print(params.file_name)
    print('Results directory')
    print(params.folder)

    print('--------------------- \n')
    


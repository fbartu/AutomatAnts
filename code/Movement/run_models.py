from Model import *

for r in np.arange(0, 1.01, 0.05):
    for R in np.arange(2.5, 20.01, 2.5):
        m = Model(rho = r, R = R)
        m.run()
        fn = 'rho_%s_R_%s' % (r, R)
        m.save_results('/home/polfer/research/gits/AutomatAnts/results/2024/movement_results/', fn)

        
        

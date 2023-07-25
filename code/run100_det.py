# import adaptation_model
import simplified_model
import json
foodXnode = 1
# adaptation_model.food = dict.fromkeys(adaptation_model.food_positions, foodXnode)
# m = adaptation_model.Model()
simplified_model.food = dict.fromkeys(simplified_model.food_positions, foodXnode)
m = simplified_model.Model()
i = 0
discarded_sims = 0
results = {}
while i < 100:
    m.run()
    if m.iters > 15000:
        t = [i[0].is_collected for i in m.food.values()]
        if sum(t) > 0:
            tlist = [i[0].collection_time for i in m.food.values() if hasattr(i[0], 'collection_time')]
            # minv, maxv = adaptation_model.np.min(tlist), adaptation_model.np.max(tlist)
            # if sum(t) == len(adaptation_model.food):
            minv, maxv = simplified_model.np.min(tlist), simplified_model.np.max(tlist)
            if sum(t) == len(simplified_model.food):
                f = True
            else:
                f = False
        # results[i] = {'T': m.T, 'N': m.N, 'I': m.I, 'gIn': m.gIn, 'gOut': m.gOut, 'Food': [minv, maxv, f]}
        # result = {'T': m.T, 'N': m.N, 'I': m.I, 'gIn': m.gIn, 'gOut': m.gOut, 'gInit': [m.agents[i].g for i in m.agents]}
        result = {'T': m.T, 'N': m.N, 'I': m.I, 'gIn': m.gIn, 'gOut': m.gOut}

        path = '../results/FINAL_SIMULATIONS/det/det_%s.json' % i
        with open (path, 'w') as f:
            json.dump(result, f)
        # df = adaptation_model.pd.DataFrame({'T': m.T, 'N': m.N, 'I': m.I, 'gIn': m.gIn, 'gOut': m.gOut})
        # df_T = adaptation_model.pd.DataFrame.transpose(df)
        
        # df_T.to_csv(path) 
        i += 1
    else:
        discarded_sims += 1
    # adaptation_model.food = dict.fromkeys(adaptation_model.food_positions, foodXnode)
    # del m
    # m = adaptation_model.Model()
    simplified_model.food = dict.fromkeys(simplified_model.food_positions, foodXnode)
    del m
    m = simplified_model.Model()
    
print('Finished with %s discarded simulations') % str(discarded_sims)
    
# df = adaptation_model.pd.DataFrame(results)
# df_T = adaptation_model.pd.DataFrame.transpose(df)
# df_T.to_csv('../results/determinist.csv')
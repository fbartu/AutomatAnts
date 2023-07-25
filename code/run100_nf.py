import simplified_model
import json
simplified_model.food = dict.fromkeys(simplified_model.food_positions, -1)
m = simplified_model.Model()
i = 0
discarded_sims = 0
results = {}
while i < 100:
    m.run()
    if m.iters > 15000:
        minv, maxv, f = simplified_model.np.nan, simplified_model.np.nan, False
        # results[i] = {'T': m.T, 'N': m.N, 'I': m.I, 'gIn': m.gIn, 'gOut': m.gOut, 'Food': [minv, maxv, f]}
        result = {'T': m.T, 'N': m.N, 'I': m.I, 'gIn': m.gIn, 'gOut': m.gOut}
        path = '../results/FINAL_SIMULATIONS/nf/nf_%s.json' % i
        with open (path, 'w') as f:
            json.dump(result, f)
        i += 1
    else:
        discarded_sims += 1
    
    del m
    m = simplified_model.Model()
print('Finished with %s discarded simulations') % str(discarded_sims)
# df = simplified_model.pd.DataFrame(results)
# df_T = simplified_model.pd.DataFrame.transpose(df)
# df_T.to_csv('../results/no_food.csv')
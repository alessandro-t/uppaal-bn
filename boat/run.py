import bnlearn
import json
import numpy as np
import os
import pandas as pd
import subprocess

from bnlearn.bnlearn import to_BayesianModel
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination

def remove_duplicates(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]

def parse_traces(simulations, seq_length):
    _, _, traces = str(simulations.stdout).partition('Formula is satisfied.')
    monitors = []
    data = {}

    idx = ' '
    while len(idx) > 0:
        monitor, idx, rest = traces.partition('[0]')
        monitor = monitor.replace('\\n', '').replace(':', '').strip()
        if len(idx):
            monitors.append(monitor)
            data[monitor] = []
            rest = rest.split('\\n')
            partial_data = rest[:seq_length]
            for trace in partial_data:
                trace = trace.split(':')[1]
                trace = np.array(eval(trace.strip().replace(' ', ',')))
                try:
                    value = trace[trace[:, 1] > 0][0]
                except:
                    value = trace[-1]
                data[monitor].append(value)

            traces = '\\n'.join(rest[seq_length:])
    for monitor in data:
        data[monitor] = np.array(data[monitor])
    return monitors, data

if __name__ == '__main__':
    verifyta_path = json.load(open('../config.json'))['verifyta_path']
    seq_length = 10000

    query = '\n'.join(open('uppaal_models/simple_model_query.xml').readlines())
    query = query.replace('seq_length', str(seq_length))
    with open('uppaal_models/tmp_simple_model_query.xml', 'wt') as tmp_query:
        tmp_query.write(query)

    simulations = subprocess.run([verifyta_path, \
                                  'uppaal_models/simple_model.xml', \
                                  'uppaal_models/tmp_simple_model_query.xml'], \
                                   shell=False, stdout=subprocess.PIPE)
    os.remove('uppaal_models/tmp_simple_model_query.xml')

    monitors, data = parse_traces(simulations, seq_length)
    new_data = []
    new_data_time = []
    # POST PROCESSING (MODEL DEPENDENT)
    unique_monitors = remove_duplicates([m.split('.')[0] for m in monitors ])
    for um in unique_monitors:
        columns = [monitor for monitor in monitors if monitor.startswith(um)]

        time = np.array([data[c][:,0] for c in columns])
        winner = np.argmin(time, axis=0)
        time = np.min(time, axis=0)
        winner = np.vectorize(lambda t:( {0:'Car', 1:'Ship'})[t])(winner)
        new_data.append(winner)
        new_data_time.append(time)

    df_data = pd.DataFrame(np.array(new_data).T, columns=unique_monitors)
    df_data_time = pd.DataFrame(np.array(new_data_time).T, columns=unique_monitors)
    # END POST PROCESSING

    model = bnlearn.structure_learning.fit(df_data, methodtype='ex', scoretype='bic', verbose=0)

    # MLE Estimation
    adjmat = model['adjmat']                                                 
    bayesian_model = to_BayesianModel(adjmat, verbose=0)                    
    mle = MaximumLikelihoodEstimator(bayesian_model, df_data)                                                                   
    cpds = []                                                            
    for node in mle.state_names:                                         
        try:                                                             
            bayesian_model.add_cpds(mle.estimate_cpd(node))                       
        except:                                                          
            pass                          
    model['model'] = bayesian_model                  
    model_infer = VariableElimination(model['model'])


    print('$p(\Phi_{END})$')
    q1 = model_infer.query(variables=['MonitorEND'], evidence={}, show_progress=0)
    print(q1)
    print('\n')

    print('$p(\Phi_{END} | \Phi_{W} = Car)$')
    q2 = model_infer.query(variables=['MonitorEND'], evidence={'MonitorW':'Car'}, show_progress=0)
    print(q2)
    print('\n')

    print('$p(\Phi_{END} | \Phi_{W} = Car, \Phi_{B} = Ship)$')
    q3 = model_infer.query(variables=['MonitorEND'], evidence={'MonitorW':'Car',
                                                               'MonitorB':'Ship'}, show_progress=0)
    print(q3)
    print('\n')

    print('$p(\Phi_{END} | \Phi_{W} = Car, \Phi_{B} = Ship, \Phi_{C} = Ship)$')
    q4 = model_infer.query(variables=['MonitorEND'], evidence={'MonitorW':'Car',
                                                               'MonitorB':'Ship',
                                                               'MonitorC':'Ship'}, show_progress=0)
    print(q4)


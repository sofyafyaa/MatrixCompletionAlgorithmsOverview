import numpy as np
import pandas as pd
from itertools import product
from tqdm import tqdm
import json
import time


class ComparisonSimulation:
    def __init__(self, simulators, const_params, varying_params):
        self.simulators = simulators
        self.const_params = const_params
        self.varying_params = varying_params

        self.results = []


    def run_sumulation(self):

        varying_params = list(self.varying_params.keys())
        varying_val = list(self.varying_params.values())
        param_grid = list(product(*varying_val))

        for sim in tqdm(self.simulators, desc="Simulators"):
            sim_name = sim.__name__

            for combo in tqdm(param_grid, desc="Parametrs grid"):
                params = self.const_params.copy()
                
                for param, value in zip(varying_params, combo):
                    params[param] = value
                
                params_str = json.dumps(params)

                simulator_it = sim(params_str)
                simulator_it.run()

                record = {
                    "simulator": sim_name,
                    "noise lvl": params.get("noise_level", np.nan),
                    "rank": params.get("rank", np.nan),
                    "missing_fraction": params.get("missing_fraction", np.nan),
                }

                specific_params = {}
                for k, v in params.items():
                    if k not in self.const_params:
                        specific_params = {k: v}

                record.update(specific_params)

                record_results = {
                    "error_history": simulator_it.error_history,
                    "time_history": simulator_it.time_history
                }

                record.update(record_results)

                self.results.append(record)

    
    def get_results(self):
        df = pd.DataFrame(self.results)

        for col in ['error_history', 'time_history']:
            if df[col].dtype == object and isinstance(df[col].iloc[0], str):
                df[col] = df[col].apply(lambda x: json.loads(x))
        
        df = df.explode(['error_history', 'time_history'])

        df = df.rename(columns={
            'error_history': 'error',
            'time_history': 'time'
        })
        
        df['iteration'] = df.groupby(list(df.columns)[:-2]).cumcount()

        return df


    def save_results(self, filename):
        df = self.get_results().copy()
        df.to_csv(filename, index=False)
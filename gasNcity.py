import pandas as pd
import numpy as np
import pickle
import itertools
from collections import OrderedDict
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import pickle
import os
from tqdm import tqdm
import pickle
import os
from scipy.optimize import root

class GasNCity():
    def __init__(self, load_models=False, folder_path = ''):
        self.qp_cols = ['QGRS_1', 'QGRS_2', 'QPlant_1', 'QPlant_2', 'QPlant_3', 'QPlant_4',
                        'PGRS_1', 'PGRS_2', 'P_1', 'P_2', 'P_3', 'P_4', 'P_5', 'P_6', 'P_7',
                        'P_8', 'P_9', 'Q_1', 'Q_2', 'Q_3', 'Q_4', 'Q_5', 'Q_6', 'Q_7']
        self.qp_models = {}
        self.valves = ['valve_1', 'valve_2', 'valve_3', 'valve_4', 'valve_5', 'valve_6',
                       'valve_7', 'valve_8', 'valve_9', 'valve_10', 'valve_11', 'valve_12']
        self.good_valves = ['valve_1', 'valve_2', 'valve_3', 
                            'valve_5', 'valve_10', 'valve_11', 'valve_12']
        self.valve_models = {}
        if load_models:
            for device in self.qp_cols:
                path = os.path.join(folder_path, f'{device}_model.sav')
                self.qp_models[device] = pickle.load(open(path, 'rb'))
            for valve in self.good_valves:
                path = os.path.join(folder_path, f'{valve}_model.sav')
                self.valve_models[valve] = pickle.load(open(path, 'rb'))
            path_mean = os.path.join(folder_path, 'means.sav')
            self.qp_mean = pickle.load(open(path_mean, 'rb'))
            path_std = os.path.join(folder_path, 'stds.sav')
            self.qp_std = pickle.load(open(path_std, 'rb'))
    
    def predict_qp(self, valves):
        valves = pd.DataFrame(valves)
        out_data = {}
        for device, model in self.qp_models.items():
            if device.startswith('QGRS') or device.startswith('PGRS'):
                out_data[device] = model.predict(valves)
        
        df_grs = pd.DataFrame(out_data)
        valves_grs = valves.join(df_grs)
        for device, model in self.qp_models.items():
            if not device.startswith('QGRS') and not device.startswith('PGRS'):
                out_data[device] = model.predict(valves_grs)
                
        out_df = pd.DataFrame(out_data)
        return out_df
    
    def predict_good_valves(self, qps):
        pqs = pd.DataFrame(qps)
        out_valves = {}
        for valve, model in self.valve_models.items():
            out_valves[valve] = model.predict(qps.values)
        out_df = pd.DataFrame(out_valves)
        return out_df
            
    
    def init_models(self):
        lr_qp_devices = self.qp_cols #['P_3', 'P_5', 'P_6', 'Q_4', 'Q_5']
        rf_qp_devices = [dev for dev in self.qp_cols if dev not in lr_qp_devices]
        
        lr_v_models = ['valve_3']
        rf_v_models = ['valve_1', 'valve_2', 'valve_5', 'valve_10', 'valve_11', 'valve_12']
        for device in lr_qp_devices:
            self.qp_models[device] = LinearRegression()
        for device in rf_qp_devices:
            self.qp_models[device] = RandomForestRegressor(n_estimators=50, max_depth=10)
        for v in lr_v_models:
            self.valve_models[v] = LinearRegression()
        for v in rf_v_models:
            self.valve_models[v] = RandomForestRegressor(n_estimators=50, max_depth=10)
            
    def fit_models(self, data: pd.DataFrame, y: pd.DataFrame, verbose=False):
        '''
        data: valve values
        y: device values
        '''
        self.qp_mean = y.mean()
        self.qp_std = y.std()
        for valve, model in self.valve_models.items():
            self.valve_models[valve] = model.fit(y, data[valve])
            if verbose:
                print(f'Model for {valve} fited')
        grs_data = data.join(y[['QGRS_1', 'QGRS_2', 'PGRS_1', 'PGRS_2']])
        for device, model in self.qp_models.items():
            if device.startswith('QGRS') or device.startswith('PGRS'):
                self.qp_models[device] = model.fit(data, y[device])
            else:
                self.qp_models[device] = model.fit(grs_data, y[device])
            if verbose:
                print(f'Model for {device} fited')
    
    
    def save_models(self, folder: str = ''):
        for device, model in self.qp_models.items():
            path = os.path.join(folder, f'{device}_model.sav')
            pickle.dump(model, open(path, 'wb'))
        for device, model in self.valve_models.items():
            path = os.path.join(folder, f'{device}_model.sav')
            pickle.dump(model, open(path, 'wb'))
        path_mean = os.path.join(folder, 'means.sav')
        self.qp_mean = pickle.dump(self.qp_mean, open(path_mean, 'wb'))
        path_std = os.path.join(folder, 'stds.sav')
        self.qp_mean = pickle.dump(self.qp_std, open(path_std, 'wb'))
    
    
    def find_valves(self, qps, constraint=[], verbose=False):
        '''Optimize valve values to fit to the qp requirements'''
        valves_to_predict = [v for v in self.valves if v not in self.good_valves and v not in constraint]
        grid = OrderedDict()
        for v in sorted(valves_to_predict, key=lambda t: int(t.split('_')[-1])):
            grid[v] = np.linspace(0.1, 1, num=6, endpoint=True)
        params = OrderedDict()
        for v in sorted(self.valves, key=lambda t: int(t.split('_')[-1])):
            params[v] = []
        out = OrderedDict()
        for v in sorted(self.valves, key=lambda t: int(t.split('_')[-1])):
            out[v] = []
        
        good_valves = np.clip(self.predict_good_valves(qps), 0, 1)
        for v in good_valves.columns:
            out[v] = good_valves[v].values
        for v in constraint:
            out[v] = np.zeros(len(qps))
        for entry, qp in enumerate(qps.values):

            best_rmse = np.inf
            best_combo = {}
            for guess in tqdm(itertools.product(*grid.values())):
                for v in good_valves.columns:
                    params[v] = [good_valves[v][entry]]
                for v in constraint:
                    params[v] = [0.]
                for i, v in enumerate(grid.keys()):
                    params[v] = [guess[i]]
                
                qp_pred = (self.predict_qp(pd.DataFrame(params)) - self.qp_mean) / self.qp_std
                qp_scaled = (qp - self.qp_mean) / self.qp_std
                rmse = np.sum(((qp_pred - qp_scaled)**2 / (qp_scaled+1e-6)**2).values)
                
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_combo = params
            
            preds = self.predict_qp(pd.DataFrame(best_combo))
            qp_pred = self.check_validity(preds)
            if not qp_pred[0][0]:    
                for v in [x for x in good_valves if x not in constraint]:
                    delta = 0.1 if best_combo[v] > 0.5 else 0.2
                    best_combo[v] = np.clip(best_combo[v]+delta, 0 , 1)
            for v in valves_to_predict:
                out[v].append(best_combo[v][0])
                                    
            
            if verbose:
                print(f'Solution found for {entry}-th entry')
        return pd.DataFrame(out)

    
    def plant_q_rule(self, series, threshold=1e-1):
        return (series >= threshold).astype(int)


    def private_q_rule(self, series, threshold=0.6):
        name = series.name
        if name in ['Q_1', 'Q_7']:
            threshold *= 2
        if name in ['Q_6']:
            threshold *= 4
        return (series >= threshold).astype(int)


    def get_q_by_p(self, pressure):
        if pressure.name in ['P_1', 'P_8']:
            return -0.3 + pressure * 7 / 1e6

        elif pressure.name in ['P_2', 'P_3', 'P_4', 'P_6']:
            return -0.2 + pressure * 4 / 1e6


    def validate_plant(self, preds):
        res = self.plant_q_rule(preds)
        return res


    def validate_private(self, preds):
        res = self.private_q_rule(preds)
        return res


    def check_distribution(self, preds):
        pass


    def check_pressure_order(self, idx, preds):
        determined_pairs = [
            ('P_9', 'P_8'),
            ('P_7', 'P_8'),
            ('P_7', 'P_6'),
            ('P_7', 'P_5'),
            ('P_4', 'P_3'),
            ('P_6', 'P_3'),
            ('P_6', 'P_2'),
        ]

        most_probable_pairs = [
            ('P_9', 'P_1'),
            ('P_7', 'P_4'),
        ]

        for pair in determined_pairs:
            if preds[pair[0]] < preds[pair[1]]:
                print(f"WARNING in row {idx}: There's no such case in the dataset: "
                      f"{pair[1]} is greater than {pair[0]}. Please check your predictions")

        for pair in most_probable_pairs:
            if preds[pair[0]] < preds[pair[1]]:
                print(f"WARNING in row {idx}: There are very few such case in the dataset: "
                      f"{pair[1]} is greater than {pair[0]}. Please check your predictions")
        return


    def check_p_q_relationship(self, preds):
        mapping = {
            'P_1': 'Q_1',
            'P_4': 'Q_4',
            'P_6': 'Q_5',
            'P_8': 'Q_7',
            'P_2': 'Q_2',
            'P_3': 'Q_3',
        }

        for i in range(1, 10):
            col = f'P_{i}'
            # If not checked, continue
            if col not in mapping:
                continue

            # Preprocess column
            series = preds[col]
            if i == 2:
                series /= 2
                series += preds['P_1'] / 2

            # Check the difference between predicted and original
            diff_allowed = 0.05
            if i in [2, 3]:
                diff_allowed = 0.07
            elif i in [1, 8]:
                diff_allowed = 0.1

            q_pred = self.get_q_by_p(series)
            diff = (preds[mapping[col]] - q_pred).abs()
            bad_index = diff[diff > diff_allowed].index
            if bad_index.shape[0] > 0:
                print(f'WARNING in rows {bad_index.values}: {col} does not follow {mapping[col]} relationship')


    def check_validity(self, preds):
        # Check "Q exceeds min level"
        for i in range(1, 8):
            preds[f'validPrivate_{i}'] = self.validate_private(preds[f'Q_{i}'])

        for i in range(1, 5):
            preds[f'validPlant_{i}'] = self.validate_plant(preds[f'QPlant_{i}'])

        # Check "P_9 > 200 000 for Plant 4"
        preds['validP9'] = (preds['P_9'] > 200000).astype(int)

        # Check order of Pressures
        for idx, row in preds.iterrows():
            self.check_pressure_order(idx, row)

        # Check that some specific Q and P follow their linear relationship
        self.check_p_q_relationship(preds)

        cols = [i for i in preds.columns if i.startswith('valid')]
        validity_score = preds[cols].mean(axis=1)
        validity_binary = preds[cols].product(axis=1)

        return validity_binary, validity_score



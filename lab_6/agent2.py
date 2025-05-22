import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

soil_moisture = ctrl.Antecedent(np.arange(0, 101, 1), 'soil_moisture')
disease = ctrl.Antecedent(np.arange(0, 5.1, 0.1), 'disease')
fertilizer_amount = ctrl.Consequent(np.arange(0, 11, 1), 'fertilizer_amount')

soil_moisture['low'] = fuzz.trimf(soil_moisture.universe, [0, 0, 40])
soil_moisture['medium'] = fuzz.trimf(soil_moisture.universe, [30, 50, 70])
soil_moisture['high'] = fuzz.trimf(soil_moisture.universe, [60, 100, 100])

disease['healthy'] = fuzz.trimf(disease.universe, [0, 0, 0.5])
disease['common_rust'] = fuzz.trimf(disease.universe, [0.5, 1, 1.5])
disease['scab'] = fuzz.trimf(disease.universe, [1.5, 2, 2.5])
disease['early_blight'] = fuzz.trimf(disease.universe, [2.5, 3, 3.5])
disease['yellow_curl'] = fuzz.trimf(disease.universe, [3.5, 4, 4.5])

fertilizer_amount['none'] = fuzz.trimf(fertilizer_amount.universe, [0, 0, 2])
fertilizer_amount['low'] = fuzz.trimf(fertilizer_amount.universe, [1, 3, 5])
fertilizer_amount['medium'] = fuzz.trimf(fertilizer_amount.universe, [4, 6, 8])
fertilizer_amount['high'] = fuzz.trimf(fertilizer_amount.universe, [7, 10, 10])

rules = [
    ctrl.Rule(soil_moisture['medium'] & disease['early_blight'], fertilizer_amount['high']),
    ctrl.Rule(soil_moisture['medium'] & disease['yellow_curl'], fertilizer_amount['medium']),
    ctrl.Rule(soil_moisture['medium'] & disease['common_rust'], fertilizer_amount['medium']),
    ctrl.Rule(soil_moisture['medium'] & disease['scab'], fertilizer_amount['low']),
    ctrl.Rule(soil_moisture['medium'] & disease['healthy'], fertilizer_amount['none']),

    ctrl.Rule(soil_moisture['high'] & disease['early_blight'], fertilizer_amount['high']),
    ctrl.Rule(soil_moisture['high'] & disease['yellow_curl'], fertilizer_amount['high']),
    ctrl.Rule(soil_moisture['high'] & disease['common_rust'], fertilizer_amount['high']),
    ctrl.Rule(soil_moisture['high'] & disease['scab'], fertilizer_amount['medium']),
    ctrl.Rule(soil_moisture['high'] & disease['healthy'], fertilizer_amount['none']),

    ctrl.Rule(soil_moisture['low'] & disease['early_blight'], fertilizer_amount['medium']),
    ctrl.Rule(soil_moisture['low'] & disease['yellow_curl'], fertilizer_amount['low']),
    ctrl.Rule(soil_moisture['low'] & disease['common_rust'], fertilizer_amount['low']),
    ctrl.Rule(soil_moisture['low'] & disease['scab'], fertilizer_amount['low']),
    ctrl.Rule(soil_moisture['low'] & disease['healthy'], fertilizer_amount['none']),
]
ctrl_system = ctrl.ControlSystem(rules)


def pred(soil, disease_name):
    disease_map = {
        "healthy": 0,
        "common_rust": 1,
        "scab": 2,
        "early_blight": 3,
        "yellow_curl": 4,
    }
    sim = ctrl.ControlSystemSimulation(ctrl_system)
    sim.input['soil_moisture'] = soil
    sim.input['disease'] = disease_map[disease_name]
    sim.compute()
    return round(sim.output['fertilizer_amount'], 2)

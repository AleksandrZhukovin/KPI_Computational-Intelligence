import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
import matplotlib


matplotlib.use('TkAgg')


temperature = ctrl.Antecedent(np.arange(15, 41, 1), 'temperature')
humidity = ctrl.Antecedent(np.arange(0, 101, 1), 'humidity')
room_volume = ctrl.Antecedent(np.arange(15, 151, 10), 'room_volume')
fan_speed = ctrl.Consequent(np.arange(0, 101, 1), 'fan_speed')

temperature['low'] = fuzz.trimf(temperature.universe, [15, 15, 25])
temperature['medium'] = fuzz.trimf(temperature.universe, [20, 25, 30])
temperature['high'] = fuzz.trimf(temperature.universe, [25, 40, 40])

humidity['low'] = fuzz.trimf(humidity.universe, [0, 0, 50])
humidity['medium'] = fuzz.trimf(humidity.universe, [25, 50, 75])
humidity['high'] = fuzz.trimf(humidity.universe, [50, 100, 100])

room_volume['low'] = fuzz.trimf(room_volume.universe, [15, 15, 50])
room_volume['medium'] = fuzz.trimf(room_volume.universe, [50, 75, 100])
room_volume['high'] = fuzz.trimf(room_volume.universe, [100, 150, 150])

fan_speed['low'] = fuzz.trimf(fan_speed.universe, [0, 0, 50])
fan_speed['medium'] = fuzz.trimf(fan_speed.universe, [25, 50, 75])
fan_speed['high'] = fuzz.trimf(fan_speed.universe, [50, 100, 100])

# temperature.view()
# humidity.view()
# room_volume.view()
# fan_speed.view()
# plt.show()

rule1 = ctrl.Rule(temperature['high'] & humidity['high'] & room_volume['high'], fan_speed['high'])
rule2 = ctrl.Rule(temperature['medium'] & humidity['high'] & room_volume['medium'], fan_speed['medium'])
rule3 = ctrl.Rule(temperature['low'] & humidity['low'] & room_volume['low'], fan_speed['low'])
rule4 = ctrl.Rule(temperature['low'] & humidity['low'] & room_volume['high'], fan_speed['high'])
rule5 = ctrl.Rule(temperature['high'] & humidity['high'] & room_volume['low'], fan_speed['medium'])

fan_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5])
fan_simulation = ctrl.ControlSystemSimulation(fan_ctrl)

fan_simulation.input['temperature'] = 40
fan_simulation.input['humidity'] = 80
fan_simulation.input['room_volume'] = 130

fan_simulation.compute()
print(fan_simulation.output['fan_speed'])

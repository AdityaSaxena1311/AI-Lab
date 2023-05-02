import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
temp = ctrl.Antecedent(np.arange(0, 101, 1), 'Temperature')
humidity = ctrl.Antecedent(np.arange(0, 101, 1), 'Humidity')
speed = ctrl.Consequent(np.arange(0, 101, 1), 'Speed')
temp['cold'] = fuzz.trimf(temp.universe, [0, 0, 50])
temp['hot'] = fuzz.trimf(temp.universe, [50, 100, 100])
humidity['dry'] = fuzz.trimf(humidity.universe, [0, 0, 50])
humidity['wet'] = fuzz.trimf(humidity.universe, [50, 100, 100])
speed['slow'] = fuzz.trimf(speed.universe, [0, 0, 50])
speed['fast'] = fuzz.trimf(speed.universe, [50, 100, 100])
rule1 = ctrl.Rule(temp['cold'] | humidity['dry'], speed['slow'])
rule2 = ctrl.Rule(temp['hot'] | humidity['wet'], speed['fast'])
rule3 = ctrl.Rule(humidity['dry'] & temp['hot'], speed['fast'])
rule4 = ctrl.Rule(humidity['wet'] & temp['cold'], speed['slow'])
speed_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4])
speed_simulation = ctrl.ControlSystemSimulation(speed_ctrl)
speed_simulation.input['Temperature'] = 30
speed_simulation.input['Humidity'] = 70
speed_simulation.compute()
speed.view(sim=speed_simulation)
plt.show()

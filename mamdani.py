import warnings

warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import numpy as np
import math

from fuzzy_expert.variable import FuzzyVariable
from fuzzy_expert.inference import DecompositionalInference

from fuzzy_expert.rule import FuzzyRule
from ipywidgets import interact, widgets


pi = math.pi #Declaração Função pi
x = np.arange(0.000001, 2*pi, 0.01) #Vetor de entrada de 0 a 2pi
y = np.sin(x)/x #vetor de saída sen(x)/x


#Declaração das variáveis Etrada (x) e Saída (y)
variables = {
    "X": FuzzyVariable(
        universe_range=(0.000001, 2*pi),
        terms={
            "A1": ('trimf', 0, 0, pi/2), #X aproximadamente 0
            "A2": ('trimf', 0, pi/2, pi), #X aproximadamente pi/2
            "A3": ('trimf', pi/2, pi, (3*pi)/2), #X aproximadamente pi
            "A4": ('trimf', pi, (3*pi)/2, 2*pi), #X aproximadamente 3pi/2
            "A5": ('trimf', (3*pi)/2, 2*pi, 2*pi), #X aproximadamente 2pi
        },step=0.01
    ),
    "Y": FuzzyVariable(
        universe_range=(-1, 1),
        terms={
            "B1": ('trimf', -0.4, -0.2, -0.1), #Y aproximadamente -0.2
            "B2": ('trimf', -0.05, 0, 0.05), #Y aproximadamente 0
            "B3": ('trimf', 0.6, 0.7, 0.8), #Y aproximadamente 0.7
            "B4": ('trimf', 0.9, 1, 1), #Y aproximadamente 1
        },
    )
    
}

#Plota gráfico de entrada
plt.figure(figsize=(10, 2.5))
variables["X"].plot()

#Plota gráfico de saída
plt.figure(figsize=(10, 2.5))
variables["Y"].plot()

#Parâmetros das funções de pertinência
rules = [
    #SE X APROX. 0, ENTÃO Y APROX. 1
    FuzzyRule(
        premise=[
            ("X", "A1")
        ],
        consequence=[("Y", "B4")],
    ),
    #SE X APROX. PI/2, ENTÃO Y APROX. 0.7
    FuzzyRule(
        premise=[
            ("X", "A2")
        ],
        consequence=[("Y", "B3")],
    ),
    #SE X APROX. PI, ENTÃO Y APROX. 0
    FuzzyRule(
         premise=[
             ("X", "A3")
         ],
         consequence=[("Y", "B2")],
     ),
    #SE X APROX. 3PI/2, ENTÃO Y APROX. -0.2
    FuzzyRule(
         premise=[
             ("X", "A4")
         ],
         consequence=[("Y", "B1")],
     ),
    #SE X APROX. 2PI, ENTÃO Y APROX. 0
    FuzzyRule(
         premise=[
             ("X", "A5")
         ],
         consequence=[("Y", "B2")],
     )
]

print(rules[0])
print()
print(rules[1])
print()
print(rules[2])
print()
print(rules[3])
print()
print(rules[4])

#Executar Modelo
model = DecompositionalInference(
    and_operator="min",
    or_operator="max",
    implication_operator="Rc",
    composition_operator="max-min",
    production_link="max",
    defuzzification_operator="cog",
)

#Plotar modelo de teste
plt.figure(figsize=(10, 6))
model.plot(
    variables=variables,
    rules=rules,
    X=5.77
)


#CI 1 Fuzzy

import numpy as np

A = np.array([0.2, 0.4, 0.6, 0.8])
B = np.array([0.5, 0.3, 0.7, 0.9])

def union(A, B): return np.maximum(A, B)
def intersection(A, B): return np.minimum(A, B)
def compliment(A): return 1-A
def difference(A, B): return np.minimum(A, 1-B)

print("union : ", union(A,B))
print("intersection : ", intersection(A,B))
print("complement : ", compliment(A))
print("difference : ", difference(A,B))

P = np.array([0.2, 0.5, 0.8])
Q = np.array([0.4, 0.6, 0.9])
R = np.array([0.3, 0.7, 1.0])

def cartesian_product(A, B):
    return np.array([[min(a,b) for b in B] for a in A])

R1 = cartesian_product(P, Q)
R2 = cartesian_product(Q, R)
print("PxQ\n", R1)
print("QXR\n", R2)

def min_max(R, S):
    rows, mid = R.shape
    cols = S.shape[1]
    result = np.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            result[i][j] = max(min(R[i][k], S[k][j]) for k in range (mid))
    return result

composition = min_max(R1, R2)
print("Composition\n", composition)

"""#CI 2 GA"""

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load iris.csv
data = pd.read_csv("iris.csv")
x = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Convert labels if needed (string → numeric)
from sklearn.preprocessing import LabelEncoder
y = LabelEncoder().fit_transform(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

def fitness(ind):
    h, lr = ind
    model = MLPClassifier(hidden_layer_sizes=(h,), learning_rate_init=lr, max_iter=200)
    model.fit(x_train, y_train)
    return 1 - accuracy_score(y_test, model.predict(x_test))

# Initialize population
pop = [[np.random.randint(5,50), np.random.uniform(0.001,0.1)] for _ in range(10)]

# GA loop
for _ in range(20):
    pop = sorted(pop, key=fitness)[:5]
    new = pop.copy()
    while len(new) < 10:
        p1, p2 = pop[np.random.randint(5)], pop[np.random.randint(5)]
        child = [np.random.choice([p1[0], p2[0]]), np.random.choice([p1[1], p2[1]])]
        if np.random.rand() < 0.1: child[0] = np.random.randint(5,50)
        if np.random.rand() < 0.1: child[1] = np.random.uniform(0.001,0.1)
        new.append(child)
    pop = new

best = min(pop, key=fitness)
print("\n\nBest parameters:", best, "\n\n")

"""#CI 3 CSA"""

import numpy as np

def f(x): return (x-2)**2
def aff(f): return 1/(1+f)

pop = np.random.uniform(-10, 10, 20)

for _ in range(50):
    fit = np.array([f(x) for x in pop])
    af = np.array([aff(fi) for fi in fit])

    sel = pop[np.argsort(af)[-10:]]
    clones = []

    for s in sel:
        c = np.repeat(s, 5)
        c = c + np.random.normal(0, 0.5 * (1 - aff(f(s))), 5)
        clones.extend(c)

    clones = np.array(clones)
    best = pop[np.argsort([f(x) for x in clones][:15])]
    new = np.random.uniform(-10, 10, 5)
    pop = np.concatenate([best, new])

best = pop[np.argmin([f(x) for x in pop])]
print("Best x :", best)

"""#CI 4 DEAP"""

!pip install deap
!pip install scoop

from deap import base, creator, tools, algorithms
from scoop import futures
import random

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("attr", random.uniform, -10, 10)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr, 1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def eval(ind): return ((ind[0]-2)**2,)
toolbox.register("evaluate", eval)

toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=1.0)
toolbox.register("select", tools.selTournament, tournsize=3)

toolbox.register("map", futures.map)  # Distributed

pop = toolbox.population(n=20)
algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.2, ngen=20, verbose=False)

best = tools.selBest(pop, 1)[0]
print("Best:", best[0])

"""#CI 5 ACO"""

import numpy as np

# Distance matrix (4 cities example)
dist = np.array([[0,2,9,10],
                 [1,0,6,4],
                 [15,7,0,8],
                 [6,3,12,0]])

n = len(dist)
pher = np.ones((n,n))
alpha, beta, evap = 1, 2, 0.5

def route_len(r):
    return sum(dist[r[i]][r[i+1]] for i in range(len(r)-1)) + dist[r[-1]][r[0]]

best = None
for _ in range(50):
    routes = []
    for _ in range(n):
        r = [np.random.randint(n)]
        while len(r) < n:
            i = r[-1]
            probs = [(pher[i][j]**alpha)*(1/(dist[i][j]+1e-6))**beta if j not in r else 0 for j in range(n)]
            probs = np.array(probs)/sum(probs)
            r.append(np.random.choice(range(n), p=probs))
        routes.append(r)

    pher *= (1-evap)
    for r in routes:
        l = route_len(r)
        for i in range(n-1):
            pher[r[i]][r[i+1]] += 1/l

    best = min(routes, key=route_len)

print("Best route:", best)
print("Distance:", route_len(best))

"""#DC 1 RPC"""

#server.py
from xmlrpc.server import SimpleXMLRPCServer

def factorial(n):
    result = 1
    for i in range(1, n + 1):
        result *= i
    return result

server = SimpleXMLRPCServer(("localhost", 8000))
print("RPC Server running on port 8000...")

server.register_function(factorial, "factorial")
server.serve_forever()

#client.py
import xmlrpc.client

proxy = xmlrpc.client.ServerProxy("http://localhost:8000/")

num = int(input("Enter number: "))
result = proxy.factorial(num)

print("Factorial =", result)

"""#DC 2 RMI"""

#server.py
from xmlrpc.server import SimpleXMLRPCServer

def concat(a, b):
    return a + b

server = SimpleXMLRPCServer(("localhost", 9000))
server.register_function(concat, "concat")
server.serve_forever()

#client.py
import xmlrpc.client

proxy = xmlrpc.client.ServerProxy("http://localhost:9000/")
print(proxy.concat("Hello ", "World"))

"""#DC 3 Round Robin"""



"""#DC 4"""



"""#DC 5 MapReduce"""

import csv

mapped = {}

# --- Map Phase ---
with open("weather.csv", "r") as file:
    reader = csv.DictReader(file)
    for row in reader:
        year = row["year"]
        temp = float(row["temperature"])

        if year not in mapped:
            mapped[year] = []
        mapped[year].append(temp)

# --- Reduce Phase (Average Temperature) ---
avg_temp = {}
for year, temps in mapped.items():
    avg_temp[year] = sum(temps) / len(temps)

# --- Find Hottest & Coolest ---
hottest = max(avg_temp, key=avg_temp.get)
coolest = min(avg_temp, key=avg_temp.get)

print("Average Temperature per Year:", avg_temp)
print("Hottest Year:", hottest)
print("Coolest Year:", coolest)
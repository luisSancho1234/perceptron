import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# ENTRADAS VIA CONSOLE
# ---------------------------------------------------------

# Número de amostras
amostras = int(input("Digite o número de amostras: "))

# Entradas X
print("\n--- Insira os valores de X ---")
x = []
for i in range(amostras):
    x1 = float(input(f"Digite x[{i+1}]1: "))
    x2 = float(input(f"Digite x[{i+1}]2: "))
    x.append([x1, x2])
x = np.array(x)

# Saídas desejadas T
print("\n--- Insira os valores de T ---")
t = []
for i in range(amostras):
    ti = float(input(f"Digite t[{i+1}] (1 ou -1): "))
    t.append([ti])
t = np.array(t)

# Variáveis do perceptron
print("\n--- Parâmetros do Perceptron ---")
limiar = float(input("Digite o limiar: "))
alfa = float(input("Digite alfa (taxa de aprendizagem): "))
max_epocas = int(input("Digite o número máximo de épocas: "))

# Pesos iniciais
print("\n--- Insira os pesos iniciais ---")
v1 = float(input("Digite o peso inicial v1: "))
v2 = float(input("Digite o peso inicial v2: "))
v = np.array([[v1], [v2]])

v0 = float(input("Digite o bias inicial v0: "))

entradas = 2

# ---------------------------------------------------------
# TREINAMENTO
# ---------------------------------------------------------
yin = np.zeros((amostras, 1))
y = np.zeros((amostras, 1))
ciclo = 0
convergiu = False

while ciclo < max_epocas:
    cont = 0

    for i in range(amostras):
        yin[i] = np.dot(x[i, :], v) + v0

        y[i] = 1.0 if yin[i] >= limiar else -1.0

        if y[i] == t[i]:
            cont += 1

        # Atualização dos pesos
        for j in range(entradas):
            v[j] = v[j] + alfa * (t[i] - y[i]) * x[i][j]

        v0 = v0 + alfa * (t[i] - y[i])

    ciclo += 1

    print("Ciclo:", ciclo)
    print("v:", v)
    print("v0:", v0)
    print("-" * 20)

    if cont == amostras:
        convergiu = True
        print(f"\nConvergiu na época: {ciclo}")
        break

if not convergiu:
    print("\nAviso: Atingiu o limite de épocas sem convergir.")

print("\n--- RESULTADOS FINAIS ---")
print("Pesos finais v =", v.T)
print("Peso final v0 =", v0)

# ---------------------------------------------------------
# GRÁFICO DA FRONTEIRA DE DECISÃO
# ---------------------------------------------------------

vx = np.linspace(np.min(x[:,0]) - 1, np.max(x[:,0]) + 1, 100)
vy = -(v0 + v[0] * vx) / v[1]

plt.plot(vx, vy, label='Fronteira de Decisão', color='black')

for i in range(amostras):
    if t[i] == 1.0:
        plt.scatter(x[i, 0], x[i, 1], marker='*', color='blue', s=120)
    else:
        plt.scatter(x[i, 0], x[i, 1], marker='v', color='red', s=120)

plt.title(f"Classificação do Perceptron (Épocas: {ciclo})")
plt.xlabel("Entrada 1")
plt.ylabel("Entrada 2")
plt.grid(True)
plt.legend()
inp = input("Deseja abrir o gráfico? (y/n)")
if(inp == 'y'):
    plt.show()

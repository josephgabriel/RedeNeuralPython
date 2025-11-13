import numpy as np
import matplotlib.pyplot as plt

plt.style.use('dark_background')
plt.rcParams['figure.figsize'] = (10, 8)

def Curva_Linear(x, w, b=0, Ruido_Grafico=0):
    return w*x + b + Ruido_Grafico*np.random.randn(x.shape[0])

# Dados iniciais
x = np.arange(-10, 30.1, 0.5)
Y = Curva_Linear(x, 1.8, 32, Ruido_Grafico=2.5)

# Visualização inicial
plt.scatter(x, Y)
plt.xlabel('°C', fontsize=20)
plt.ylabel('°F', fontsize=20)

# Inicialização
w = np.random.rand(1)
b = 0

def Feedforward(inputs, w, b):
    return w * inputs + b

def mse(Y, y):
    return (Y - y) ** 2

def Backpropagation(inputs, outputs, targets, w, b, lr):
    dw = lr * (-2 * inputs * (targets - outputs)).mean()
    db = lr * (-2 * (targets - outputs)).mean()
    w -= dw
    b -= db
    return w, b

def Ajustar_Modelo(inputs, target, w, b, epochs=200, lr=0.001):
    for epoch in range(epochs):
        outputs = Feedforward(inputs, w, b)
        perda = np.mean(mse(target, outputs))
        w, b = Backpropagation(inputs, outputs, target, w, b, lr)

        if (epoch + 1) % 50 == 0:
            print(f'Epoch: [{(epoch + 1)}/{epochs}] Perda: [{perda:.4f}]')

    return w, b

# Gera dados para treino
x = np.arange(-10, 10, 2)
Y = Curva_Linear(x, w=1.8, b=32, Ruido_Grafico=1)

# Inicializa pesos
w = np.random.randn(1)
b = np.zeros(1)

# Treina o modelo
w, b = Ajustar_Modelo(x, Y, w, b, epochs=2000, lr=0.003)
print(f'w: {w[0]:.3f}, b: {b[0]:.3f}')

# Plot e salvamento
plt.figure(figsize=(10, 8))
plt.scatter(x, Y, label='Dados reais', color='cyan')
plt.plot(x, Curva_Linear(x, w, b), 'r', lw=3, label='Linha ajustada')
plt.xlabel('°C', fontsize=20)
plt.ylabel('°F', fontsize=20)
plt.title('Regressão Linear aprendida do zero', fontsize=18)
plt.legend()

# Salva o gráfico gerado
plt.savefig('resultado_regressao.png', dpi=300, bbox_inches='tight')
print("✅ Gráfico salvo como 'resultado_regressao.png'")

plt.show()

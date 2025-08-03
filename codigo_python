# -*- coding: utf-8 -*-
import numpy as np
from PIL import Image
from scipy.signal import convolve2d
import os

def apply_magma_colormap(v):
    """
    Mapeia um valor de 0 a 1 para uma cor no estilo "magma" (preto->vermelho->amarelo).
    Isso cria um visual mais vibrante e científico.
    """
    # Garante que v esteja no intervalo [0, 1]
    v = np.clip(v, 0, 1)

    # Define os pontos de cor do gradiente
    # (ponto, (R, G, B))
    colors = [
        (0.0, (0, 0, 0)),      # Preto
        (0.25, (130, 20, 90)),  # Roxo escuro
        (0.5, (250, 80, 40)),   # Laranja avermelhado
        (0.75, (250, 200, 50)), # Amarelo
        (1.0, (252, 252, 252))  # Branco/Amarelo claro
    ]

    # Encontra o segmento correto e interpola a cor
    for i in range(len(colors) - 1):
        p1, c1 = colors[i]
        p2, c2 = colors[i+1]
        if p1 <= v <= p2:
            # Interpolação linear
            t = (v - p1) / (p2 - p1)
            r = int(c1[0] + t * (c2[0] - c1[0]))
            g = int(c1[1] + t * (c2[1] - c1[1]))
            b = int(c1[2] + t * (c2[2] - c1[2]))
            return (r, g, b)
    return colors[-1][1]


class ReactionDiffusionSimulator:
    """
    Simula sistemas de reação-difusão 2D (modelo Gray-Scott).
    Versão corrigida para gerar uma única imagem final de alta qualidade.
    """
    def __init__(self, size=256, Du=0.16, Dv=0.08, feed=0.0367, kill=0.0649):
        """Inicializa o simulador com os parâmetros."""
        self.size = size
        self.Du, self.Dv = Du, Dv
        self.feed = feed
        self.kill = kill

        # Inicialização da grade
        self.U = np.ones((size, size))
        self.V = np.zeros((size, size))

        # Perturbação inicial aleatória para um padrão mais orgânico
        np.random.seed(42) # Semente para resultados reproduzíveis
        r = size // 8
        center = size // 2
        self.V[center-r:center+r, center-r:center+r] = np.random.rand(r*2, r*2)

        self.laplacian_kernel = np.array([[0.05, 0.2, 0.05],
                                          [0.2,  -1,  0.2 ],
                                          [0.05, 0.2, 0.05]])

        print(f"Simulador CORRIGIDO inicializado com F={self.feed}, k={self.kill}")

    def _update(self, dt=1.0):
        """Executa um passo da simulação."""
        lap_U = convolve2d(self.U, self.laplacian_kernel, mode='same', boundary='wrap')
        lap_V = convolve2d(self.V, self.laplacian_kernel, mode='same', boundary='wrap')

        reaction_term = self.U * self.V**2

        dU = (self.Du * lap_U - reaction_term + self.feed * (1 - self.U)) * dt
        dV = (self.Dv * lap_V + reaction_term - (self.feed + self.kill) * self.V) * dt

        self.U += dU
        self.V += dV

    def run_and_save(self, steps, filename="padrao_gerado.png"):
        """Executa a simulação e salva UMA ÚNICA imagem final."""
        print("Iniciando simulação... Isso pode levar alguns segundos.")
        for step in range(steps):
            for _ in range(3): # 3 atualizações por passo
                self._update()

            if step > 0 and step % 1000 == 0:
                print(f"Progresso: {step}/{steps} passos concluídos.")

        print("Simulação concluída. Gerando imagem final...")

        v_norm = np.clip(self.V, 0, 1)

        image_array = np.zeros((self.size, self.size, 3), dtype=np.uint8)
        for r in range(self.size):
            for c in range(self.size):
                image_array[r, c] = apply_magma_colormap(v_norm[r, c])

        img = Image.fromarray(image_array, 'RGB')
        img.save(filename)
        print(f"Feito! Imagem salva como '{filename}'.")

# --- Exemplo de Uso ---
if __name__ == '__main__':

    # Parâmetros para um padrão de leopardo
    leopard_params = {
        'feed': 0.035,
        'kill': 0.060
    }

    # Simulação do padrão de leopardo
    simulator_leopard = ReactionDiffusionSimulator(size=256, **leopard_params)
    simulator_leopard.run_and_save(steps=8000, filename="padrao_leopardo.png")

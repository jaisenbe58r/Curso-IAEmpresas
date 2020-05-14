# -*- coding: utf-8 -*-

# Inteligencia Artificial Aplicada a Negocios y Empresas
# Parte 1 - Optimizacion de los flujos de trabajo en un almacen con Q-Learning

# Importacion de las librerias
import numpy as np
import argparse
import sys

# PARTE 1 - DEFINICIoN DEL ENTORNO

# Definicion de los estados
location_to_state = {'A': 0,
                     'B': 1,
                     'C': 2,
                     'D': 3,
                     'E': 4,
                     'F': 5,
                     'G': 6, 
                     'H': 7, 
                     'I': 8,
                     'J': 9,
                     'K': 10,
                     'L': 11}

# Definicion de las acciones
actions = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

# Definicion de las recompensas
# Columnas:    A,B,C,D,E,F,G,H,I,J,K,L
R = np.array([[0,1,0,0,0,0,0,0,0,0,0,0], # A
              [1,0,1,0,0,1,0,0,0,0,0,0], # B
              [0,1,0,0,0,0,1,0,0,0,0,0], # C
              [0,0,0,0,0,0,0,1,0,0,0,0], # D
              [0,0,0,0,0,0,0,0,1,0,0,0], # E
              [0,1,0,0,0,0,0,0,0,1,0,0], # F
              [0,0,1,0,0,0,1,1,0,0,0,0], # G
              [0,0,0,1,0,0,1,0,0,0,0,1], # H
              [0,0,0,0,1,0,0,0,0,1,0,0], # I
              [0,0,0,0,0,1,0,0,1,0,1,0], # J
              [0,0,0,0,0,0,0,0,0,1,0,1], # K
              [0,0,0,0,0,0,0,1,0,0,1,0]])# L

# PARTE 2 - CONSTRUCCIoN DE LA SOLUCIoN DE IA CON Q-LEARNING

# Transformacion inversa de estados a ubicaciones
state_to_location = {state : location for location, state in location_to_state.items()}

# Crear la funcion final que nos devuelva la ruta optima
def route(args, starting_location, ending_location):
    R_new = np.copy(R)
    ending_state = location_to_state[ending_location]
    R_new[ending_state, ending_state] = 1000

    Q = np.array(np.zeros([12, 12]))
    for i in range(args.epochs):
        current_state = np.random.randint(0, 12)
        playable_actions = []
        for j in range(12):
            if R_new[current_state, j] > 0:
                playable_actions.append(j)
        next_state = np.random.choice(playable_actions)
        _TD = R_new[current_state, next_state] + args.gamma*Q[next_state, np.argmax(Q[next_state,])] - Q[current_state, next_state]
        Q[current_state, next_state] = Q[current_state, next_state] + args.alpha*_TD

    route = [starting_location]
    next_location = starting_location
    while(next_location != ending_location):
        starting_state = location_to_state[starting_location]
        next_state = np.argmax(Q[starting_state, ])
        next_location = state_to_location[next_state]
        route.append(next_location)
        starting_location = next_location
    return route

# PARTE 3 - PONER EL MODELO EN PRODUCCIoN
def best_route(args, starting_location, intermediary_location, ending_location):
    return route(args, starting_location, intermediary_location) + route(args, intermediary_location, ending_location)[1:]

def arguments():
    # Parseador de argumentos
    argparser = argparse.ArgumentParser(sys.argv, description="Parseador de argumentos")

    argparser.add_argument(
        "--gamma",
        type=float,
        default=0.75,
        help="Factor de descuento [0, 1]",
    )
    argparser.add_argument(
        "--alpha",
        type=float,
        default=0.9,
        help="Tasa de aprendizaje [0, 1]",
    )
    argparser.add_argument(
        "--epochs",
        type=int,
        default=1000,
        help="Iteraciones para el aprendizaje del proceso de Q-Learning",
    )
    args = argparser.parse_args()

    return args

if __name__ == "__main__":

    args = arguments()

    # Imprimir la ruta final
    print("Ruta Elegida:")
    print(best_route(args, 'E', 'B', 'G'))
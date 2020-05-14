# -*- coding: utf-8 -*-

"""Jaime Sendra Berenguer-2018-2022.
Inteligencia Artificial Aplicada a Negocios y Empresas -- Profesor: Juan Gabriel Gomila
Estudiante: Jaime Sendra Berenguer <www.linkedin.com/in/jaisenbe>
"""

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

class Qlearning(object):
    """Class QLearning object.

        Parameters
        ----------
        starting_location: Starting point of the route.
        ending_location: End point of the route.
    
    """
    def __init__(self, R, location_to_state, actions, priority=None, negative=None, gamma=0.75, alpha=0.1, epochs=1000):
        """Init class.

        Parameters
        ----------
        R: Reward Matrix.
        location_to_state: convert location to state
        actions: Actions space
        priority: Priority points on the path
        negative: Negative points on the path
        gamma: Discount factor
        alpha: learning rate
        epochs: Iterations for learning the Q-Learning process

        """
        self.R_obective = 1000
        self.R_priority = self.R_obective*0.1
        self.R_negative = (-1)*self.R_obective

        if not isinstance(R, np.ndarray):
            raise TypeError("Invalid type {}".format(type(R)))
        else:
            self.R_initial = R
            self.R = np.copy(self.R_initial)

        if not isinstance(location_to_state, dict):
            raise TypeError("Invalid type {}".format(type(location_to_state)))
        else:
            self.location_to_state = location_to_state

        if not isinstance(actions, list):
            raise TypeError("Invalid type {}".format(type(actions)))
        else:
            self.actions = actions

        if not isinstance(gamma, float):
            raise TypeError("Invalid type {}".format(type(gamma)))
        else:
            if gamma>0.0 and gamma<=1.0:
                self.gamma = args.gamma
            else:
                raise NameError("the value must be between (0, 1]")
 
        if not isinstance(alpha, float):
            raise TypeError("Invalid type {}".format(type(alpha)))
        else:
            if alpha>0.0 and alpha<=1.0:
                self.alpha = args.alpha
            else:
                raise NameError("the value must be between (0, 1]")

        if not isinstance(epochs, int):
            raise TypeError("Invalid type {}".format(type(epochs)))
        else:
            if epochs>=0:
                self.epochs = args.epochs
            else:
                raise NameError("the value must be greater than zero")

        # Transformacion inversa de estados a ubicaciones
        self.state_to_location = {state : location for location, state in self.location_to_state.items()}

        if priority is not None:
            if isinstance(priority, list) or isinstance(priority, tuple):
                if len(priority)>0:
                    self.priority = priority[0]
                    # Priority points
                    for i in range(len(self.priority)):
                        _priority = self.location_to_state[self.priority[i]]
                        self.R[_priority, _priority] = self.R_priority
                else:
                    raise NameError("You must enter a position")
            else:
                raise TypeError("Invalid type {}".format(type(priority)))

        if negative is not None:
            if isinstance(negative, list) or isinstance(negative, tuple):
                if len(negative)>0:
                    self.negative = negative[0]
                    # Negative points
                    for i in range(len(self.negative)):
                        _negative = self.location_to_state[self.negative[i]]
                        self.R[_negative, _negative] = self.R_negative
                else:
                    raise NameError("You must enter a position")
            else:
                raise TypeError("Invalid type {}".format(type(negative)))

    def route(self, starting_location, ending_location):
        """Crear la funcion final que nos devuelva la ruta optima.

        Parameters
        ----------
        starting_location: Starting point of the route.
        ending_location: End point of the route.
            
        """
        R_new = np.copy(self.R)
        ending_state = self.location_to_state[ending_location]
        R_new[ending_state, ending_state] = self.R_obective

        Q = np.array(np.zeros([self.R.shape[0], self.R.shape[1]]))
        for i in range(self.epochs):
            current_state = np.random.randint(0, len(self.actions))
            playable_actions = []
            for j in range(len(self.location_to_state)):
                if R_new[current_state, j] > 0:
                    playable_actions.append(j)
                    
            next_state = np.random.choice(playable_actions)
            _TD = R_new[current_state, next_state] + self.gamma*Q[next_state, np.argmax(Q[next_state,])] - Q[current_state, next_state]
            Q[current_state, next_state] = Q[current_state, next_state] + args.alpha*_TD

        route = [starting_location]
        next_location = starting_location

        while(next_location != ending_location):
            starting_state = self.location_to_state[starting_location]
            next_state = np.argmax(Q[starting_state, ])
            next_location = self.state_to_location[next_state]
            route.append(next_location)
            starting_location = next_location
        return route

    # PARTE 3 - PONER EL MODELO EN PRODUCCIoN

    def best_route(self, starting_location, intermediary_location, ending_location):
        return self.route(starting_location, intermediary_location) + self.route(intermediary_location, ending_location)[1:]

    def multi_route(self, location=None):
        """
        Location refers to all the locations over witch you want pass.
            i.e. location = ['E', 'L', 'A, 'D']
        """
        if location is not None:
            if isinstance(location, list) or isinstance(location, tuple):
                if len(location)>1:
                    self.location = location
                else:
                    raise NameError("You must enter a start position and an end position")
            else:
                raise TypeError("Invalid type {}".format(type(location)))
        else:
            raise NameError("You must enter a start position and an end position")

        route = [self.location[0]]
        for i in range(1, len(location)):
            location[i]
            aux_route = self.route(location[i-1], location[i])
            route += aux_route[1:]

        return route

    def front_end(self, args):
        # Imprimir puntos de inicio y fin
        if args.location is not None:
            if isinstance(args.location[0], list) or isinstance(args.location[0], tuple):
                if not len(args.location[0])>1:
                    raise NameError("You must enter a start position and an end position")
            else:
                raise TypeError("Invalid type {}".format(type(args.location[0])))
        else:
            raise NameError("You must enter a start position and an end position")

        try:
            print("Puntos de paso obligado por orden: ", args.location[0])
        except NameError:
            print("You must enter a start position and an end position")

        # Imprimir pasos prioritarios de paso
        try:
            print("Puntos de paso prioritario: ", args.priority[0])
        except:
            print("Sin puntos de paso prioritario en la ruta")
        # Imprimir pasos negativos de paso
        try:
            print("Puntos de paso negativo: ", args.negative[0])
        except:
            print("Sin puntos de paso negativo en la ruta")
        # Imprimir la ruta final
        print("Ruta Elegida: ", self.multi_route(args.location[0]))


def arguments():
    # Parseador de argumentos
    argparser = argparse.ArgumentParser(sys.argv, description="Parseador de argumentos")

    argparser.add_argument(
        "--gamma",
        type=float,
        default=0.8,
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
        default=100000,
        help="Iteraciones para el aprendizaje del proceso de Q-Learning",
    )
    argparser.add_argument(
        "--location",
        type=str,
        action='append',
        nargs='+', 
        help="Listado de puntos de paso obligatorio.",
    )
    argparser.add_argument(
        "--priority",
        type=str,
        action='append',
        nargs='+', 
        help="Listado de puntos de interés positivo, es decir, el robot siempre que pueda se obligará a pasar por estos puntos.",
    )
    argparser.add_argument(
        "--negative",
        type=str,
        action='append',
        nargs='+', 
        help="Listado de puntos de interés negativo, es decir, el robot siempre que pueda intentar desviar la trayectoria para no pasar por estos puntos.",
    )
    args = argparser.parse_args()

    return args

if __name__ == "__main__":

    args = arguments()

    # Objeto Q-Learning
    route = Qlearning(R, location_to_state, actions, priority=args.priority, negative=args.negative, gamma=args.gamma, alpha=args.alpha, epochs=args.epochs)
    # resultados
    route.front_end(args)

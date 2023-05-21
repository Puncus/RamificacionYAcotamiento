from typing import List
import numpy as np
from scipy.optimize import linprog, OptimizeResult
from dsplot.tree import BinaryTree


def subscript(number):
    subscript_map = {
        "0": "₀",
        "1": "₁",
        "2": "₂",
        "3": "₃",
        "4": "₄",
        "5": "₅",
        "6": "₆",
        "7": "₇",
        "8": "₈",
        "9": "₉",
        "i": "ᵢ",
        }

    return ''.join(subscript_map[digit] for digit in str(number))

def gen_labels(amount):
    labels = []
    for i in range(amount):
        labels.append(f"x{subscript(f'{i+1}')}")
    return labels

class Planteamiento:
    def __init__(self, funcion_objetivo: List, restricciones_desigualdad: List, restricciones_igualdad: List = [], 
                 modo = "Maximizar", tipo="entero", variables_enteras: List= [], variables_continuas: List = []):
        if tipo not in ["Entero", "Mixto", "Binario"]:
            raise Exception(f"{modo} No es un tipo de problema valido.")
        self.no_enteras_label.setVisible(False)
        self.no_enteras_line.setVisible(False)
        self.modo = "Minimizar"
        self.tipo = tipo
        self.funcion_objetivo = np.array(funcion_objetivo)
        self.variables_enteras = variables_enteras
        self.variables_continuas = variables_continuas
        

        # El solver utilizado minimiza por default, si queremos maximizar hay que multiplicar la funcion objetivo por -1
        if modo == "Maximizar":
            self.modo = "Maximizar"
            self.funcion_objetivo *= -1
        
        self.restricciones_desigualdad = np.array([restriccion[:-1] for restriccion in restricciones_desigualdad])
        self.valor_restricciones_desigualdad = np.array([valor[-1] for valor in restricciones_desigualdad])
        
        if restricciones_igualdad:
            self.restricciones_igualdad = np.array([restriccion[:-1] for restriccion in restricciones_igualdad])
            self.valor_restricciones_igualdad = np.array([valor[-1] for valor in restricciones_igualdad])
        else:
            self.restricciones_igualdad = []
            self.valor_restricciones_igualdad = []
        
        self.solucion: OptimizeResult or None = None
        
        if tipo == "Binario":
            numero_variables = len(self.funcion_objetivo)
            for i in range(numero_variables):
                temp = [0.0] * numero_variables
                temp[i] = 1.0
                self.restricciones_desigualdad = np.concatenate((self.restricciones_desigualdad, np.array([temp])), axis=0)
                self.valor_restricciones_desigualdad = np.concatenate((self.valor_restricciones_desigualdad, np.array([1.0])), axis=0)


    def __str__(self) -> str:
        modelo = ""
        
        # Formato a la función objetivo
        if self.modo == "Minimizar":
            modelo += "Minimizar\nz = "
            for i, x in enumerate(self.funcion_objetivo, start=1):
                modelo += f"{x}x{subscript(i)}"
                try:
                    if x >= 0:
                        modelo += "+"
                except IndexError:
                    continue
        else:
            modelo += "Maximizar\nz = "
            for i, x in enumerate(self.funcion_objetivo, start=1):
                modelo += f"{-x}x{subscript(i)}"
                try:
                    if x >= 0:
                        modelo += "+"
                except IndexError:
                    continue
        
        modelo += "\n\nSujeto a:\n\n"

        # Formato a las restricciones
        for j, restriccion in enumerate(self.restricciones_desigualdad):
            for i, x in enumerate(restriccion, start=1):
                modelo += f"{x}x{subscript(i)}"
                try:
                    if restriccion[i] >= 0:
                        modelo += "+"
                except IndexError:
                    continue
            modelo += f" <= {self.valor_restricciones_desigualdad[j]}\n"
            
        if self.tipo == "Entero":
            modelo += f"x{subscript('i')} ∈ ℤ"

        elif self.tipo == "Mixto":
            
            modelo += "x"
            for var in self.variables_continuas:
                modelo += f"{subscript(var)},"
            modelo += "\b ∈ ℝ\n"

            modelo += "x"
            for var in self.variables_enteras:
                modelo += f"{subscript(var)},"
            modelo += "\b ∈ ℤ"

        elif self.tipo == "Binario":
            modelo += f"x{subscript('i')} ∈ " + "{0,1}"
        return modelo
    

    def solve(self):
        if self.restricciones_igualdad:
            self.solucion = linprog(self.funcion_objetivo, A_ub=self.restricciones_desigualdad, b_ub=self.valor_restricciones_desigualdad,
                                     A_eq=self.restricciones_igualdad, b_eq=self.valor_restricciones_igualdad)
        else:
            self.solucion = linprog(self.funcion_objetivo, A_ub=self.restricciones_desigualdad, b_ub=self.valor_restricciones_desigualdad)

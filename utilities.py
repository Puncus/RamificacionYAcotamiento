from typing import List, Tuple
import numpy as np
from queue import Queue
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


# Clase que implementa los problemas de ramificación y acotamiento 
class Planteamiento:
    def __init__(self, funcion_objetivo: List, restricciones_desigualdad: List, restricciones_igualdad: List = [], 
                 modo = "Maximizar", tipo="Entero", variables_enteras: List= [], variables_continuas: List = []):
        if tipo not in ["Entero", "Mixto", "Binario"]:
            raise Exception(f"{modo} No es un tipo de problema valido.")
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
                    if self.funcion_objetivo[i] >= 0:
                        modelo += "+"
                except IndexError:
                    continue
        else:
            modelo += "Maximizar\nz = "
            for i, x in enumerate(self.funcion_objetivo, start=1):
                modelo += f"{-x}x{subscript(i)}"
                try:
                    if self.funcion_objetivo[i] >= 0:
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


def is_approx_integer(float_number, tolerance=1e-6):
    absolute_difference = abs(float_number - round(float_number))
    return absolute_difference < tolerance


class Nodo:
    def __init__(self, funcion_objetivo, restricciones:dict, modo) -> None:
        self.right: Nodo or None = None
        self.left: Nodo or None = None
        self.restricciones: dict = restricciones
        self.ramificado = False
        self.modo = modo
        if len(self.restricciones["igualdades"]) > 0:
            self.solucion: OptimizeResult = linprog(funcion_objetivo,
                                                    A_ub=self.restricciones["desigualdades"],
                                                    b_ub=self.restricciones["objetivo_desigualdades"],
                                                    A_eq=self.restricciones["igualdades"],
                                                    b_eq=self.restricciones["objetivo_igualdades"])
        else:
            self.solucion: OptimizeResult = linprog(funcion_objetivo,
                                                    A_ub=self.restricciones["desigualdades"],
                                                    b_ub=self.restricciones["objetivo_desigualdades"])

    def __str__(self) -> str:
        resultado = f"Variables: \n"
        for index, variable in enumerate(self.solucion.x, start=1):
            if is_approx_integer(variable, tolerance=1e-8):
                resultado += f"x{subscript(f'{index}')} = {int(variable)}\n"
            else:
                resultado += f"x{subscript(f'{index}')} = {variable}\n"
        if self.modo == "Maximizar":
            resultado += f"\nZ: {abs(self.solucion.fun)}"
        else:
            resultado += f"\nZ: {self.solucion.fun}"
        return resultado


class Arbol:
    def __init__(self, raiz: Nodo) -> None:
        self.raiz = raiz
        print(raiz)

    def validar_restricciones(self, nodo: Nodo) -> Tuple[bool, str]:
        if nodo.problema.tipo == "Entero":
            for variable in nodo.problema.solucion.x:
                if is_approx_integer(variable):
                    continue
                else:
                    return False, "aún hay reales en la solución"
            return True, "Solución factible"
    def ramificar(self, nodo):
        valor = self.validar_restricciones(nodo)
        print(valor[1])

    def recorrer(self):
        cola_nodos: Queue = Queue()
        orden_nodos = []

        if self.raiz is not None:
            orden_nodos.append(self.raiz)
            cola_nodos.put(self.raiz)

        while not cola_nodos.empty():
            nodo_actual: Nodo = cola_nodos.get()

            if nodo_actual.right is not None:
                orden_nodos.append(nodo_actual.right)
                cola_nodos.put(nodo_actual.right)

            else:
                orden_nodos.append(None)

            if nodo_actual.left is not None:
                cola_nodos.put(nodo_actual.left)
                orden_nodos.append(nodo_actual.left)

            else:
                orden_nodos.append(None)
        
        return orden_nodos


    def representar(self):
        nodos: List[Nodo] = self.recorrer()
        representacion = []
        for nodo in nodos:
            if nodo is not None:
                representacion.append(f"{nodo}")
            else:
                representacion.append("None")
        
        for item in representacion:
            print(f"{item}: {type(item)}")

        image_representation = BinaryTree(nodes=representacion)
        image_representation.plot(orientation="LR", shape="circle", fill_color="#346eeb")
        from PIL import Image
        img = Image.open(f"tree.png")
        img.show()

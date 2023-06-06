from math import floor, ceil
from typing import List, Tuple
import numpy as np
from queue import Queue
from scipy.optimize import linprog, OptimizeResult
from dsplot.tree import BinaryTree
from copy import deepcopy


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
    def __init__(self, funcion_objetivo, restricciones:dict, modo, tipo, variables_continuas:list | None = None) -> None:
        self.parent: Nodo = None
        self.right: Nodo or None = None
        self.left: Nodo or None = None
        self.funcion_objetivo = funcion_objetivo
        self.restricciones: dict = restricciones
        self.ramificado = False
        self.modo = modo
        self.tipo = tipo
        self.variables_continuas = variables_continuas
        
        if len(self.restricciones["igualdades"]) > 0:
            self.solucion: OptimizeResult = linprog(self.funcion_objetivo,
                                                    A_ub=self.restricciones["desigualdades"],
                                                    b_ub=self.restricciones["objetivo_desigualdades"],
                                                    A_eq=self.restricciones["igualdades"],
                                                    b_eq=self.restricciones["objetivo_igualdades"])
        else:
            self.solucion: OptimizeResult = linprog(self.funcion_objetivo,
                                                    A_ub=self.restricciones["desigualdades"],
                                                    b_ub=self.restricciones["objetivo_desigualdades"])
           
        if self.tipo == "Entero":
            self.ramificar = staticmethod(self.ramificar_entero)
        elif self.tipo == "Binario":
            self.ramificar = staticmethod(self.ramificar_binario)
        elif self.tipo == "Mixto":
            print("si entra")
            self.ramificar = staticmethod(self.ramificar_mixto)


    def __str__(self) -> str:
        resultado = f"Variables: \n"
        try:
            for index, variable in enumerate(self.solucion.x, start=1):
                if is_approx_integer(variable, tolerance=1e-8):
                    resultado += f"x{subscript(f'{index}')} = {int(variable)}\n"
                else:
                    resultado += f"x{subscript(f'{index}')} = {variable}\n"
            if self.modo == "Maximizar":
                resultado += f"\nZ: {abs(self.solucion.fun)}"
            else:
                resultado += f"\nZ: {self.solucion.fun}"
        except TypeError:
            return "Solución no factible \nRamificación terminada"
        return resultado
    
    def ramificar_binario(self):
        print("entra a ramificar binario")
        print(self.restricciones)
        print(self.modo)
        print(self.tipo)
        restricciones_cumplen = True
        for variable in self.solucion.x:
            if variable != 1 and variable != 0:
                restricciones_cumplen = False
        if restricciones_cumplen:
            return
        
        # print(f"{self.solucion.x} aún tiene reales que deberían ser enteros")

        if self.parent is not None:    
            if self.modo == "Maximizar":
                if abs(self.solucion.fun) >= abs(self.parent.solucion.fun):
                    print(f"Problema Agotado")
                    return
            if self.modo == "Minimizar":
                if self.solucion.fun <=self.parent.solucion.fun:
                    print(f"Problema Agotado")
                    return
                
        for indice, variable in enumerate(self.solucion.x):

            if variable == 1 or variable == 0:
                continue
            else:
                # Parte derecha
                restricciones_derecha = deepcopy(self.restricciones)
                nueva_restriccion_derecha = np.zeros(len(self.funcion_objetivo))
                nueva_restriccion_derecha[indice] = 1.0 #[0.0, 0.0, ..., 1.0, 0.0, ..., 0.0]

                if len(restricciones_derecha["igualdades"]) == 0:
                    restricciones_derecha["igualdades"] = np.array([nueva_restriccion_derecha])
                else:
                    restricciones_derecha["igualdades"] = np.concatenate((restricciones_derecha["igualdades"], np.array([nueva_restriccion_derecha])), axis=0)

                if len(restricciones_derecha["objetivo_igualdades"]) == 0:
                    restricciones_derecha["objetivo_igualdades"] = np.array([1.0])
                else:
                    restricciones_derecha["objetivo_igualdades"] = np.concatenate(
                                                                    (restricciones_derecha["objetivo_igualdades"],np.array([1.0])), axis=0)
                
                # Parte Izquierda
                restricciones_izquierda = deepcopy(self.restricciones)
                nueva_restriccion_izquierda = np.zeros(len(self.funcion_objetivo))
                nueva_restriccion_izquierda[indice] = 1.0 #[0.0, 0.0, ..., 1.0, 0.0, ..., 0.0]
                
                if len(restricciones_izquierda["igualdades"]) == 0:
                    restricciones_izquierda["igualdades"] = np.array([nueva_restriccion_izquierda])
                else:
                    restricciones_izquierda["igualdades"] = np.concatenate((restricciones_izquierda["igualdades"], np.array([nueva_restriccion_izquierda])), axis=0)

                if len(restricciones_izquierda["objetivo_igualdades"]) == 0:
                    restricciones_izquierda["objetivo_igualdades"] = np.array([0.0])
                else:
                    restricciones_izquierda["objetivo_igualdades"] = np.concatenate(
                                                                    (restricciones_izquierda["objetivo_igualdades"],np.array([0.0])), axis=0)
                print(f"Restricciones izquierda:\n{restricciones_izquierda}\n\nRestricciones derecha:\n{restricciones_derecha}")

                self.right = Nodo(self.funcion_objetivo, restricciones_derecha, self.modo, self.tipo)
                self.right.parent = self
                if self.right.solucion.x is None:
                    self.right = None

                self.left = Nodo(self.funcion_objetivo, restricciones_izquierda, self.modo, self.tipo)
                self.left.parent = self
                if self.left.solucion.x is None:
                    self.left = None

                return
        indice = variable = None
        self.left = None
        self.right = None

    def ramificar_mixto(self):
        restricciones_cumplen = True
        print(self.variables_continuas)
        for index, variable in enumerate(self.solucion.x):
            if not is_approx_integer(variable) and (index in self.variables_continuas):
                restricciones_cumplen = False
        if restricciones_cumplen:
            return
        
        # print(f"{self.solucion.x} aún tienen reales en la solución")

        if self.parent is not None:    
            if self.modo == "Maximizar":
                if abs(self.solucion.fun) >= abs(self.parent.solucion.fun):
                    print(f"Problema Agotado")
                    return
            if self.modo == "Minimizar":
                if self.solucion.fun <=self.parent.solucion.fun:
                    print(f"Problema Agotado")
                    return
            
        for indice, variable in enumerate(self.solucion.x):

            if not is_approx_integer(variable) and (indice in self.variables_continuas):
                # Parte derecha
                restricciones_derecha = deepcopy(self.restricciones)
                nueva_restriccion_derecha = np.zeros(len(self.funcion_objetivo))
                nueva_restriccion_derecha[indice] = -1.0 #[0.0, 0.0, ..., 1.0, 0.0, ..., 0.0]
                restricciones_derecha["desigualdades"] = np.concatenate(
                                                            (restricciones_derecha["desigualdades"],np.array([nueva_restriccion_derecha]))
                                                            , axis=0)
                restricciones_derecha["objetivo_desigualdades"] = np.concatenate(
                                                            (restricciones_derecha["objetivo_desigualdades"],np.array([-ceil(variable)]))
                                                            , axis=0)
                
                # Parte Izquierda
                restricciones_izquierda = deepcopy(self.restricciones)
                nueva_restriccion_izquierda = np.zeros(len(self.funcion_objetivo))
                nueva_restriccion_izquierda[indice] = 1.0 #[0.0, 0.0, ..., 1.0, 0.0, ..., 0.0]
                restricciones_izquierda["desigualdades"] = np.concatenate(
                                                            (restricciones_izquierda["desigualdades"],np.array([nueva_restriccion_izquierda]))
                                                            , axis=0)
                restricciones_izquierda["objetivo_desigualdades"] = np.concatenate(
                                                            (restricciones_izquierda["objetivo_desigualdades"],np.array([floor(variable)]))
                                                            , axis=0)

                self.right = Nodo(self.funcion_objetivo, restricciones_derecha, self.modo, self.tipo, self.variables_continuas)
                self.right.parent = self
                if self.right.solucion.x is None:
                    self.right = None

                self.left = Nodo(self.funcion_objetivo, restricciones_izquierda, self.modo, self.tipo, self.variables_continuas)
                self.left.parent = self
                if self.left.solucion.x is None:
                    self.left = None

                return
            else:
                continue
        indice = variable = None
        self.left = None
        self.right = None

    
    def ramificar_entero(self):
        restricciones_cumplen = True
        for variable in self.solucion.x:
            if not is_approx_integer(variable):
                restricciones_cumplen = False
        if restricciones_cumplen:
            return
        
        # print(f"{self.solucion.x} aún tienen reales en la solución")

        if self.parent is not None:    
            if self.modo == "Maximizar":
                if abs(self.solucion.fun) >= abs(self.parent.solucion.fun):
                    print(f"Problema Agotado")
                    return
            if self.modo == "Minimizar":
                if self.solucion.fun <=self.parent.solucion.fun:
                    print(f"Problema Agotado")
                    return
            
        for indice, variable in enumerate(self.solucion.x):

            if is_approx_integer(variable):
                continue
            else:
                # Parte derecha
                restricciones_derecha = deepcopy(self.restricciones)
                nueva_restriccion_derecha = np.zeros(len(self.funcion_objetivo))
                nueva_restriccion_derecha[indice] = -1.0 #[0.0, 0.0, ..., 1.0, 0.0, ..., 0.0]
                restricciones_derecha["desigualdades"] = np.concatenate(
                                                            (restricciones_derecha["desigualdades"],np.array([nueva_restriccion_derecha]))
                                                            , axis=0)
                restricciones_derecha["objetivo_desigualdades"] = np.concatenate(
                                                            (restricciones_derecha["objetivo_desigualdades"],np.array([-ceil(variable)]))
                                                            , axis=0)
                
                # Parte Izquierda
                restricciones_izquierda = deepcopy(self.restricciones)
                nueva_restriccion_izquierda = np.zeros(len(self.funcion_objetivo))
                nueva_restriccion_izquierda[indice] = 1.0 #[0.0, 0.0, ..., 1.0, 0.0, ..., 0.0]
                restricciones_izquierda["desigualdades"] = np.concatenate(
                                                            (restricciones_izquierda["desigualdades"],np.array([nueva_restriccion_izquierda]))
                                                            , axis=0)
                restricciones_izquierda["objetivo_desigualdades"] = np.concatenate(
                                                            (restricciones_izquierda["objetivo_desigualdades"],np.array([floor(variable)]))
                                                            , axis=0)

                self.right = Nodo(self.funcion_objetivo, restricciones_derecha, self.modo, self.tipo)
                self.right.parent = self
                if self.right.solucion.x is None:
                    self.right = None

                self.left = Nodo(self.funcion_objetivo, restricciones_izquierda, self.modo, self.tipo)
                self.left.parent = self
                if self.left.solucion.x is None:
                    self.left = None

                return
        indice = variable = None
        self.left = None
        self.right = None


class Arbol:

    def solve(self):
        cola_nodos: Queue = Queue()

        if self.raiz is not None and not self.raiz.ramificado:
            cola_nodos.put(self.raiz)

        while not cola_nodos.empty():
            nodo_actual: Nodo = cola_nodos.get()
            if (not nodo_actual.ramificado) and (nodo_actual.solucion.x is not None):
                nodo_actual.ramificar()

                if (nodo_actual.right is not None) and (nodo_actual.right.solucion.x is not None):
                    cola_nodos.put(nodo_actual.right)

                if (nodo_actual.left is not None) and (nodo_actual.left.solucion.x is not None):
                    cola_nodos.put(nodo_actual.left)


    def __init__(self, raiz: Nodo) -> None:
        self.raiz = raiz
        self.solve()



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

    def recorrer_2(self):
        from treelib import Node, Tree
        cont = 1
        cola_nodos: Queue = Queue()
        prueba = Tree()

        if self.raiz is not None:
            cola_nodos.put(self.raiz)

        while not cola_nodos.empty():
            nodo_actual: Nodo = cola_nodos.get()
            if nodo_actual.parent is None:
                prueba.create_node(f"{nodo_actual.solucion.x}", nodo_actual)
            else:
                prueba.create_node(f"{nodo_actual.solucion.x}", nodo_actual, parent=nodo_actual.parent)

            if nodo_actual.left is not None:
                cola_nodos.put(nodo_actual.left)
            else:
                prueba.create_node(f"termina_ramificacion_{cont}", parent=nodo_actual)
                cont += 1
            
            if nodo_actual.right is not None:
                cola_nodos.put(nodo_actual.right)
            else:
                prueba.create_node(f"termina_ramificacion_{cont}", parent=nodo_actual)
                cont += 1
                
            

        return prueba

    def representar(self):
        nodos: List[Nodo] = self.recorrer()
        representacion = []
        for nodo in nodos:
            if nodo is not None:
                representacion.append(f"{nodo}")
            else:
                representacion.append(".")

        image_representation = BinaryTree(nodes=representacion)
        image_representation.plot(orientation="LR", shape="circle", fill_color="#346eeb")
        from PIL import Image
        img = Image.open(f"tree.png")
        img.show()

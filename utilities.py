from math import floor, ceil
from typing import List, Tuple
import numpy as np
from queue import Queue
from scipy.optimize import linprog, OptimizeResult
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
    def __init__(self, funcion_objetivo, restricciones:dict, modo, tipo, variables_continuas:list | None = None, tag: str | None = None) -> None:
        self.parent: Nodo = None
        self.right: Nodo or None = None
        self.left: Nodo or None = None
        self.funcion_objetivo = funcion_objetivo
        self.restricciones: dict = restricciones
        self.ramificado = False
        self.modo = modo
        self.tipo = tipo
        self.variables_continuas = variables_continuas
        if not tag:
            self.tag = "Solución Inicial"
        else:
            self.tag = tag
        
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
            self.solucion_factible = staticmethod(self.solucion_entera)
        elif self.tipo == "Binario":
            self.ramificar = staticmethod(self.ramificar_binario)
            self.solucion_factible = staticmethod(self.solucion_binaria)
        elif self.tipo == "Mixto":
            self.ramificar = staticmethod(self.ramificar_mixto)
            self.solucion_factible = staticmethod(self.solucion_mixta)


    def __str__(self) -> str:
        resultado = f"{self.tag}\n\nVariables: \n"
        try:
            for index, variable in enumerate(self.solucion.x, start=1):
                if is_approx_integer(variable, tolerance=1e-8):
                    resultado += f"X{index} = {int(variable)}\n"
                else:
                    resultado += f"X{index} = {variable}\n"
            if self.modo == "Maximizar":
                resultado += f"\nZ: {abs(self.solucion.fun)}"
            else:
                resultado += f"\nZ: {self.solucion.fun}"

            if self.solucion_factible():
                resultado += "\nZcota"
        except TypeError:
            return "Solución no factible \nRamificación terminada"
        return resultado
    

    def solucion_entera(self) -> bool:
        for variable in self.solucion.x:
            if not is_approx_integer(variable):
                return False
        return True


    def solucion_binaria(self) -> bool:
        for variable in self.solucion.x:
            if variable != 1 and variable != 0:
                return False
        return True
    

    def solucion_mixta(self) -> bool:
        for index, variable in enumerate(self.solucion.x):
            if not is_approx_integer(variable) and (index not in self.variables_continuas):
                return False
        return True


    def ramificar_binario(self):
        print("entra a ramificar binario")
        print(self.restricciones)
        print(self.modo)
        print(self.tipo)
        
        if self.solucion_binaria():
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
                tag_derecha = f"X{(indice + 1)} = 1"
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
                tag_izquierda = f"X{(indice + 1)} = 0"
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

                self.right = Nodo(self.funcion_objetivo, restricciones_derecha, self.modo, self.tipo, tag=tag_derecha)
                self.right.parent = self
                if self.right.solucion.x is None:
                    self.right = None

                self.left = Nodo(self.funcion_objetivo, restricciones_izquierda, self.modo, self.tipo, tag=tag_izquierda)
                self.left.parent = self
                if self.left.solucion.x is None:
                    self.left = None

                return
        indice = variable = None
        self.left = None
        self.right = None

    def ramificar_mixto(self):
        
        print(self.variables_continuas)
        if self.solucion_mixta():
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

            if not is_approx_integer(variable) and (indice not in self.variables_continuas):
                # Parte derecha
                tag_derecha = f"X{(indice + 1)} >= {[ceil(variable)]}"
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
                tag_izquierda = f"X{(indice + 1)} <= {[floor(variable)]}"
                restricciones_izquierda = deepcopy(self.restricciones)
                nueva_restriccion_izquierda = np.zeros(len(self.funcion_objetivo))
                nueva_restriccion_izquierda[indice] = 1.0 #[0.0, 0.0, ..., 1.0, 0.0, ..., 0.0]
                restricciones_izquierda["desigualdades"] = np.concatenate(
                                                            (restricciones_izquierda["desigualdades"],np.array([nueva_restriccion_izquierda]))
                                                            , axis=0)
                restricciones_izquierda["objetivo_desigualdades"] = np.concatenate(
                                                            (restricciones_izquierda["objetivo_desigualdades"],np.array([floor(variable)]))
                                                            , axis=0)

                self.right = Nodo(self.funcion_objetivo, restricciones_derecha, self.modo, self.tipo, self.variables_continuas, tag=tag_derecha)
                self.right.parent = self
                if self.right.solucion.x is None:
                    self.right = None

                self.left = Nodo(self.funcion_objetivo, restricciones_izquierda, self.modo, self.tipo, self.variables_continuas, tag=tag_izquierda)
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
        if self.solucion_entera():
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
                tag_derecha = f"X{(indice + 1)} >= {[ceil(variable)]}"
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
                tag_izquierda = f"X{(indice + 1)} <= {[floor(variable)]}"
                restricciones_izquierda = deepcopy(self.restricciones)
                nueva_restriccion_izquierda = np.zeros(len(self.funcion_objetivo))
                nueva_restriccion_izquierda[indice] = 1.0 #[0.0, 0.0, ..., 1.0, 0.0, ..., 0.0]
                restricciones_izquierda["desigualdades"] = np.concatenate(
                                                            (restricciones_izquierda["desigualdades"],np.array([nueva_restriccion_izquierda]))
                                                            , axis=0)
                restricciones_izquierda["objetivo_desigualdades"] = np.concatenate(
                                                            (restricciones_izquierda["objetivo_desigualdades"],np.array([floor(variable)]))
                                                            , axis=0)

                self.right = Nodo(self.funcion_objetivo, restricciones_derecha, self.modo, self.tipo, tag=tag_derecha)
                self.right.parent = self
                if self.right.solucion.x is None:
                    self.right = None

                self.left = Nodo(self.funcion_objetivo, restricciones_izquierda, self.modo, self.tipo, tag=tag_izquierda)
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
                prueba.create_node(f"{nodo_actual.solucion.x}, Z = {nodo_actual.solucion.fun} {nodo_actual.tag}", nodo_actual)
            else:
                prueba.create_node(f"{nodo_actual.solucion.x}Z = {nodo_actual.solucion.fun} {nodo_actual.tag}", nodo_actual, parent=nodo_actual.parent)

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

    def recorrer_3(self):
        from bigtree import Node, print_tree, tree_to_dot
        cont = 1
        cola_nodos: Queue = Queue()
        data_dict = {}
        orden = []
        raiz: Node

        if self.raiz is not None:
            cola_nodos.put(self.raiz)
            modo = self.raiz.modo

        while not cola_nodos.empty():
            nodo_actual: Nodo = cola_nodos.get()
            
            if nodo_actual.parent is None:
                raiz = Node(f"{nodo_actual.tag}", data=f"{nodo_actual.solucion.x}, Z = {nodo_actual.solucion.fun} (Solución Inicial)")
                data_dict[nodo_actual.tag] = raiz
            else:
                nodo_temp = Node(f"{nodo_actual.tag}", data=f"{nodo_actual.solucion.x}, Z = {nodo_actual.solucion.fun}", parent=data_dict[nodo_actual.parent.tag])
                data_dict[nodo_actual.tag] = deepcopy(nodo_temp)

            if nodo_actual.left is not None:
                cola_nodos.put(nodo_actual.left)
            else:
                nodo_temp = Node(f"Termina ramificacion{cont}", data=f"Sin solucion", parent=data_dict[(nodo_actual.tag)])
                data_dict[f"Termina ramificacion{cont}"] = deepcopy(nodo_temp)
                cont += 1
            
            if nodo_actual.right is not None:
                cola_nodos.put(nodo_actual.right)
            else:
                nodo_temp = Node(f"Termina ramificacion{cont}", data=f"Sin solucion", parent=data_dict[(nodo_actual.tag)])
                data_dict[f"Termina ramificacion{cont}"] = deepcopy(nodo_temp)
                cont += 1
        print(data_dict)
        print_tree(raiz, attr_list=["data"])
        representacion = tree_to_dot(raiz)
        representacion.write_png("sol.png")
        representacion.write_dot("sol.dot")

    
    def recorrer_4(self):
        from bigtree import dataframe_to_tree_by_relation, print_tree, tree_to_dot
        import pandas as pd
        import pydot as pdt
        optimos = []
        cola_nodos: Queue = Queue()
        tree_relation = []
        cont = 1
        if self.raiz is not None:
            cola_nodos.put(self.raiz)
            modo = self.raiz.modo

        while not cola_nodos.empty():
            nodo_actual: Nodo = cola_nodos.get()
            
            # Nodo actual
            if nodo_actual.parent is None:
                tree_relation.append([str(nodo_actual), None, ""])
                if nodo_actual.solucion_factible():
                    optimos.append(abs(nodo_actual.solucion.fun))
            else:
                tree_relation.append([str(nodo_actual), f"{nodo_actual.parent}", ""])
                if nodo_actual.solucion_factible():
                    optimos.append(abs(nodo_actual.solucion.fun))
            
            # Hijos nodo actual
            if nodo_actual.left is not None:
                cola_nodos.put(nodo_actual.left)
            else:
                tree_relation.append([f"fin ramificacion{cont}", str(nodo_actual), ""])
                cont += 1
            
            if nodo_actual.right is not None:
                cola_nodos.put(nodo_actual.right)
            else:
                tree_relation.append([f"fin ramificacion{cont}", str(nodo_actual), ""])
                cont += 1
        
        if modo == "Maximizar":
            optimo = max(optimos)
        else:
            optimo = min(optimos)
        datos_arbol = pd.DataFrame(tree_relation, columns=["child", "parent", "data"])
        # print(datos_arbol)
        arbol = dataframe_to_tree_by_relation(datos_arbol, child_col="child", parent_col="parent")
        print_tree(arbol)
        representacion = tree_to_dot(arbol, node_shape="circle", node_colour="white")
        dot_string = representacion.to_string()
        # print(dot_string)
        new_dot = ""
        for line in dot_string.splitlines():
            
            if ("fillcolor" in line) and ("ramificacion" not in line) and f"Z: {optimo}" not in line:
                continue
            if "fin ramificacion" in line:
                continue
            else:
                if "Zcota0" in line:
                    line = line.replace("Zcota0", "Problema agotado")
                new_dot += f"{line}\n"

        # print(new_dot)
        representacion = pdt.graph_from_dot_data(new_dot)[0]
        representacion.write_png("solucion.png")
        # representacion.write_dot("sol.dot")


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


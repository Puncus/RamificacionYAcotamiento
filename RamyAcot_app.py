from PyQt5 import uic
from utilities import gen_labels, Nodo, Arbol
from PyQt5.QtCore import QSize, Qt, QRect
from PyQt5.QtWidgets import (
    QApplication,
    QLabel,
    QLineEdit,
    QMainWindow,
    QTableWidgetItem,
    QMessageBox
)
from PyQt5.QtGui import QPixmap
from screeninfo import get_monitors
from numpy import array, concatenate


MAXIMIZAR = "Maximizar"
MINIMIZAR = "Minimizar"
MAIN_MONITOR = None

for monitor in get_monitors():
    if monitor.is_primary:
        MAIN_MONITOR = monitor


class QImage(QLabel):
    def __init__(self, image_path:str, image_name:str):
        super().__init__(image_name)
        self.setPixmap(QPixmap(image_path))


# Subclass QMainWindow to customize your application's main window
class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        uic.loadUi("app.ui", self)
        self.labels_valor_restricciones = ["Tipo", "Valor"]
        self.modo:str | None = None
        self.tipo:str | None = None

        # Conexiones para los eventos de los botones
        self.numero_variables_slider.valueChanged.connect(self.valor_variables)
        self.numero_restricciones_slider.valueChanged.connect(self.valor_restricciones)
        self.nuevo_planteamiento_action.triggered.connect(self.clear)
        self.ejemplo_action.triggered.connect(self.mostrar_ejemplo)
        self.tipo_button_group.buttonClicked.connect(self.tipo_button_pressed)
        self.modo_button_group.buttonClicked.connect(self.modo_button_pressed)
        self.resolver_boton.clicked.connect(self.solve)

        
    def tipo_button_pressed(self, button):
        self.tipo = button.text()    


    def modo_button_pressed(self, button):
        self.modo = button.text()


    def valor_variables(self):
        valor = self.numero_variables_slider.value()
        self.valor_slider_variables.setText(f"{valor}")
        self.variables_tabla.setColumnCount(valor)
        self.restricciones_tabla.setColumnCount(valor + 2)
        new_labels = gen_labels(valor)
        self.variables_tabla.setHorizontalHeaderLabels(new_labels)
        self.restricciones_tabla.setHorizontalHeaderLabels(new_labels + self.labels_valor_restricciones)
            

    def valor_restricciones(self):
        valor = self.numero_restricciones_slider.value()
        self.valor_slider_restricciones.setText(f"{valor}")
        self.restricciones_tabla.setRowCount(valor)
    

    def mostrar_ejemplo(self):
        # self.variables_line.setText("[3.0, 1.0, 3.0]")
        self.tipo = "Entero"
        self.modo = "Maximizar"
        self.tipo_entero.setChecked(True)
        self.modo_maximizar.setChecked(True)
        self.numero_variables_slider.setValue(3)
        self.numero_restricciones_slider.setValue(3)
        funcion_objetivo_ejemplo = [3.0, 1.0, 3.0]

        restricciones_ejemplo = [[-1.0, 2.0, 1.0, "<=", 4.0],
                                [0.0, 4.0, -3.0, "<=", 2.0],
                                [1.0, -3.0, 2.0, "<=", 3.0]]
        for column in range(len(funcion_objetivo_ejemplo)):
            self.variables_tabla.setItem(0, column, QTableWidgetItem(f"{funcion_objetivo_ejemplo[column]}"))
        for row in range(len(restricciones_ejemplo)):
            for column in range(len(restricciones_ejemplo[0])):
                self.restricciones_tabla.setItem(row, column, QTableWidgetItem(f"{restricciones_ejemplo[row][column]}"))


    def clear(self):
        self.tipo_entero.setChecked(False)
        self.tipo_mixto.setChecked(False)
        self.tipo_binario.setChecked(False)
        self.numero_variables_slider.setValue(2)
        self.numero_restricciones_slider.setValue(1)
        self.restricciones_tabla.setRowCount(self.numero_restricciones_slider.value())
        self.restricciones_tabla.setColumnCount(self.numero_restricciones_slider.value())
        self.variables_tabla.clear()
        self.restricciones_tabla.clear()
    

    def get_variables_from_tabla(self) -> list or None:
        data = []
        for column in range(self.variables_tabla.columnCount()):
            if self.variables_tabla.columnCount() == 1:
                return None
            item = self.variables_tabla.item(0, column)
            if item is not None:
                try:
                    data.append(float(item.text()))
                except ValueError:
                    popup = QMessageBox()
                    popup.setWindowTitle("Valores no aceptados")
                    popup.setText("Por favor ingresa valores válidos en las celdas.")
                    popup.setIcon(QMessageBox.Critical)
                    _ = popup.exec_()
                    return None
            else:
                data.append(0.0)
        return array(data)


    def get_restricciones_from_tabla(self) -> tuple or None:
        table_data = []

        for row in range(self.restricciones_tabla.rowCount()):
            row_data = []
            if self.restricciones_tabla.columnCount() == 1:
                return None
            
            for column in range(self.restricciones_tabla.columnCount()):
                item = self.restricciones_tabla.item(row, column).text()
                if item is not None:
                    if item in (["=", "<=", ">="]): 
                        row_data.append(item)    
                    else:
                        try:
                            row_data.append(float(item))
                        except ValueError:
                            print(f"{item} causó el error en {row}, {column}")
                            popup = QMessageBox()
                            popup.setWindowTitle("Valores no aceptados")
                            popup.setText("Por favor ingresa valores válidos en las celdas.")
                            popup.setIcon(QMessageBox.Critical)
                            error = popup.exec_()
                            return None
                else:
                    row_data.append(0.0)
            table_data.append(row_data)

        desigualdad_valores = []
        desigualdad_objetivos = []
        igualdad_valores = []
        igualdad_objetivos = []
        for row in table_data:
            if row[-2] == "=":
                igualdad_valores.append(row[:-2])
                igualdad_objetivos.append(row[-1])
            elif row[-2] == ">=":
                desigualdad_valores.append([value * -1 for value in row[:-2]])
                desigualdad_objetivos.append(row[-1] * -1)
            elif row[-2] == "<=":
                desigualdad_valores.append(row[:-2])
                desigualdad_objetivos.append(row[-1])
                
        restricciones = {"desigualdades":array(desigualdad_valores),
                            "objetivo_desigualdades": array(desigualdad_objetivos),
                            "igualdades": array(igualdad_valores),
                            "objetivo_igualdades": array(igualdad_objetivos)}
        return restricciones


    def solve(self):
        # print(self.tipo)
        # print(self.modo)

        funcion_objetivo = self.get_variables_from_tabla()
        if self.modo == MAXIMIZAR:
            funcion_objetivo *= -1

        restricciones = self.get_restricciones_from_tabla()
        if self.tipo == "Binario":
            numero_variables = len(funcion_objetivo)
            for i in range(numero_variables):
                temp = [0.0] * numero_variables
                temp[i] = 1.0
                restricciones["desigualdades"] = concatenate((restricciones["desigualdades"], array([temp])), axis=0)
                restricciones["objetivo_desigualdades"] = concatenate((restricciones["objetivo_desigualdades"], array([1.0])), axis=0)

        arbol_ramificacion = Arbol(Nodo(funcion_objetivo , restricciones, self.modo))
        arbol_ramificacion.representar()


# You need one (and only one) QApplication instance per application.
# Pass in sys.argv to allow command line arguments for your app.
# If you know you won't use command line arguments QApplication([]) works too.

app = QApplication([])
app.setStyle('Fusion')
# Create a Qt widget, which will be our window.
window = MainWindow()
window.show()  # IMPORTANT!!!!! Windows are hidden by default.

# Start the event loop.
app.exec()
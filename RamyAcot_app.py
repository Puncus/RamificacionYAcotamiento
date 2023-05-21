from PyQt5 import uic
from utilities import gen_labels
from PyQt5.QtCore import QSize, Qt
from PyQt5.QtWidgets import (
    QApplication,
    QLabel,
    QMainWindow,
    QTableWidgetItem
)
from PyQt5.QtGui import QPixmap

from screeninfo import get_monitors

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
        self.numero_variables_slider.valueChanged.connect(self.valor_variables)
        self.numero_restricciones_slider.valueChanged.connect(self.valor_restricciones)
        self.nuevo_planteamiento_action.triggered.connect(self.clear)
        self.ejemplo_action.triggered.connect(self.mostrar_ejemplo)
        self.labels_valor_restricciones = ["Tipo", "Valor"]
        self.resolver_boton.clicked.connect(self.solve)


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
        self.tipo_entero.setChecked(True)
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
    
    def get_table_data(self):
        restricciones = ["=", "<=", ">="]
        self.data = []
        for row in range(self.restricciones_tabla.rowCount()):
            row_data = []
            for column in range(self.restricciones_tabla.columnCount()):
                item = self.restricciones_tabla.item(row, column).text()
                if item is not None:
                    if item in restricciones: 
                        row_data.append(item)    
                    else:
                        row_data.append(float(item))
                else:
                    row_data.append(0.0)
            self.data.append(row_data)
        print(self.data)

    def solve(self):
        self.get_table_data()


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
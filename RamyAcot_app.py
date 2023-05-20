from PyQt5 import uic
from PyQt5.QtCore import QSize, Qt
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFontComboBox,
    QLabel,
    QLineEdit,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QTimeEdit,
    QVBoxLayout,
    QWidget
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
        self.ejemplo_action.triggered.connect(self.mostrar_ejemplo)
        
        
        # self.setWindowTitle("Ramificación y Acotamiento")
        # self.setFixedSize(QSize(1920, 1080))
        # self.layout = QVBoxLayout()
        # self.layout.addWidget(QLabel("Tipo de Problema"))

    
        # widgets = [
        #     QRadioButton,
        #     QRadioButton,
        #     QRadioButton
        # ]

        # for w in widgets:
        #     if w == QRadioButton:
        #         self.layout.addWidget(w("coso"))
                
        #     else:
        #         self.layout.addWidget(w())

        # widget = QWidget()
        # widget.setLayout(self.layout)
        # Set the central widget of the Window. Widget will expand
        # to take up all the space in the window by default.
        # self.setCentralWidget(widget)
    def valor_variables(self):
            valor = self.numero_variables_slider.value()
            self.valor_slider_variables.setText(f"{valor}")

    def valor_restricciones(self):
            valor = self.numero_restricciones_slider.value()
            self.valor_slider_restricciones.setText(f"{valor}")
    
    def mostrar_ejemplo(self):
         self.variables_line.setText("[3.0, 1.0, 3.0]")


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
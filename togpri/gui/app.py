# togpri/gui/app.py
"""Entrypoint de la aplicación."""
import sys
from PyQt6.QtWidgets import QApplication
from togpri.gui.main_window import MainWindow


def run():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    run()
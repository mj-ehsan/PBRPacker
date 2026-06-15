import sys

from PyQt5.QtGui import QSurfaceFormat
from PyQt5.QtWidgets import QApplication

from pack_worker import PackWorker
from pbr_gui import MainWindow


if __name__ == "__main__":
    fmt = QSurfaceFormat()
    fmt.setVersion(4, 5)
    fmt.setProfile(QSurfaceFormat.CoreProfile)
    fmt.setSwapBehavior(QSurfaceFormat.DoubleBuffer)
    fmt.setDepthBufferSize(24)
    fmt.setSamples(8)
    QSurfaceFormat.setDefaultFormat(fmt)
    
    app = QApplication(sys.argv)
    window = MainWindow(PackWorker)
    window.show()
    sys.exit(app.exec_())

'''
draw SH w/drawArrays
'''
from rendererGL import RendererGL
from PyQt5.QtWidgets import QApplication, QMainWindow, QListWidget, QGridLayout
from PyQt5.QtCore import Qt
from SphericalHarmonicsMetal import SphericalHarmonicsMetal
import random, sys, ctypes
from array import array
import numpy as np


class OpenGLWidget(RendererGL):
    def __init__(self, win, sh):
        super(OpenGLWidget, self).__init__()
        self.setFocusPolicy(Qt.StrongFocus)
        self.win, self.sh = win, sh

    def init(self, gl):
        def init_draw():
            def np2array(np_vect):  # array = numpy, take the fast lane
                arr = array('f')
                arr.frombytes(np_vect.tobytes())
                return arr

            gl.glEnableClientState(gl.GL_VERTEX_ARRAY)  # define draw array components
            gl.glEnableClientState(gl.GL_COLOR_ARRAY)
            gl.glEnableClientState(gl.GL_NORMAL_ARRAY)

            self.n_coords = self.sh.coords.size  # x+y+z
            self.n_vertex = self.n_coords / 3  # xyz

            gl.glVertexPointer(3, gl.GL_FLOAT, 0, np2array(self.sh.coords))  # array = numpy
            gl.glColorPointer(3, gl.GL_FLOAT, 0, np2array(self.sh.colors))
            gl.glNormalPointer(gl.GL_FLOAT, 0, np2array(self.sh.normals))

        self.sceneInit(gl)

        gl.glCullFace(gl.GL_FRONT)
        init_draw()

    def draw(self, gl):
        def drawSH(gl):   gl.glDrawArrays(gl.GL_TRIANGLE_FAN, 0, self.n_vertex)

        gl.glEnable(gl.GL_RESCALE_NORMAL)
        self.scale(gl, 0.1)

        drawSH(gl)


class Main(QMainWindow):
    def __init__(self, sh):
        super(Main, self).__init__()

        self.setWindowTitle(f'Spherical Harmonics({sh.resolution:}), code:{sh.code:}, color map:{sh.color_map:}')
        self.setCentralWidget(OpenGLWidget(self, sh))
        self.show()

    def create_list(self):
        self.listwidget = QListWidget()
        self.listwidget.insertItem(0, "Red")
        self.listwidget.insertItem(1, "Orange")
        self.listwidget.insertItem(2, "Blue")
        self.listwidget.insertItem(3, "White")
        self.listwidget.insertItem(4, "Green")
        # self.listwidget.clicked.connect(self.clicked)
        self.layout.addChildWidget(self.listwidget)


if __name__ == '__main__':
    from sh_codes import spherical_harmonics_codes

    print('generating METAL spherical harmonics...', end='')
    sh = SphericalHarmonicsMetal(resolution=512, code=random.choice(spherical_harmonics_codes), color_map=2)
    print('done, generating model...')

    app = QApplication(sys.argv)
    Main(sh)

    print('done')
    app.exec_()

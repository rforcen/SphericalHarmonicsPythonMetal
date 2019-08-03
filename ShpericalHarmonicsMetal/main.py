'''
    main SH drawing app. uses appl metal kernel
'''

import sys

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QApplication, QMainWindow)

from rendererGL import RendererGL
import random
from SphericalHarmonicsMetal import SphericalHarmonicsMetal


class OpenGLWidget(RendererGL):
    coords, colors, normals, texture = None, None, None, None
    sh, win = None, None
    need_compile = True
    gl_compiled_list = 1

    def __init__(self, win, sh):
        super(OpenGLWidget, self).__init__()
        self.win, self.sh = win, sh

        self.setFocusPolicy(Qt.StrongFocus)  # accepts key events

    def init(self, gl):
        self.sceneInit(gl)
        gl.glCullFace(gl.GL_FRONT)

    def draw(self, gl):
        def draw_solid(gl):
            if self.sh.coords is not None:
                gl.glEnable(gl.GL_NORMALIZE)

                gl.glBegin(gl.GL_QUADS)
                for ic in range(0, len(self.sh.coords) - (self.sh.resolution + 1)):  # tranverse quads
                    for i in (ic, ic + self.sh.resolution, ic + self.sh.resolution + 1, ic + 1):  # draw the quad CCW
                        gl.glNormal3fv(list(self.sh.normals[i]))
                        gl.glColor3fv(list(self.sh.colors[i]))
                        gl.glVertex3fv(list(self.sh.coords[i]))
                        # gl.glTexCoord2fv(list(self.sh.textures[i]))

                gl.glEnd()

        def draw_points(gl):
            if self.sh.coords is not None:
                gl.glEnable(gl.GL_NORMALIZE)
                gl.glBegin(gl.GL_POINTS)
                for ic in range(0, len(self.sh.coords)):  # tranverse quads
                    gl.glVertex3fv(list(self.sh.coords[ic]))
                    gl.glNormal3fv(list(self.sh.normals[ic]))
                    gl.glColor3fv(list(self.sh.colors[ic]))
                    # gl.glColor3fv((1,1,1))
                gl.glEnd()

        def compile_list(gl):
            if self.need_compile:
                gl.glNewList(self.gl_compiled_list, gl.GL_COMPILE)

                draw_solid(gl)
                # draw_points(gl)

                gl.glEndList()
                self.need_compile = False

        def draw_list(gl):
            compile_list(gl)
            gl.glCallList(self.gl_compiled_list)

        sc = 0.1
        gl.glScalef(sc, sc, sc)
        draw_list(gl)


class Main(QMainWindow):
    def __init__(self, sh):
        super(Main, self).__init__()

        self.setWindowTitle(f'Spherical Harmonics, code:{sh.code:}, color map:{sh.color_map:}')
        self.setCentralWidget(OpenGLWidget(self, sh))
        self.show()


if __name__ == '__main__':
    from sh_codes import spherical_harmonics_codes

    sh = SphericalHarmonicsMetal(resolution=256, code=random.choice(spherical_harmonics_codes), color_map=2)

    app = QApplication(sys.argv)
    Main(sh)

    app.exec_()

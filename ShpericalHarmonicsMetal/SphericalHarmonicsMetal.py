'''
python interface to metal app.
'''
import runmetal
import numpy as np


class SphericalHarmonicsMetal():
    '''
    // match struct in cpu XYZ->float3=4 x float32
    typedef struct { // sizeof(float3) == 16 (4 x float32), sizeof(float2)==8 (2 x float32)
        XYZ coords,  normals;
        Texture textures;
        Color colors;
    } Vertex; // sizeof(Vertex)==64
    '''

    def __init__(self, resolution, code, color_map):
        def code_2_list(code):
            return list(map(float, str(code)))

        self.pm = runmetal.PyMetal()
        self.pm.opendevice()
        self.pm.openlibrary(filename='SphericalHarmonicsVertex.metal') # file
        self.fn = self.pm.getfn("sphericalHarmonicsVertex") # func. name

        self.resolution, self.color_map, self.code = resolution, color_map, code
        res2 = resolution * resolution

        sizeofVertex = 64  # in bytes = 64 floats = (4+4+4+4) x 4 = 16 x 4 -> all aligned to 32 bits
        floats_in_vertex = int(sizeofVertex / 4)

        quad_buff = self.pm.emptybuffer(res2 * sizeofVertex)

        self.run_metal(quad_buff,
                       [quad_buff, self.pm.intBuffer(resolution), self.pm.floatBuffer(code_2_list(code)),
                        self.pm.intBuffer(color_map)])

        # generate the mesh of vertex
        mesh = self.pm.buf2numpy(quad_buff, dtype=np.float32).reshape(res2, floats_in_vertex)

        self.coords = mesh[..., 0:3]
        self.normals = mesh[..., 4:7]  # as coords is float3 == 4 x float (32 bit alignment)
        self.textures = mesh[..., 8:10]
        self.colors = mesh[..., 12:15]

    def get_vertex(self):
        return self.coords, self.normals, self.textures, self.colors

    def run_metal(self, pixelbuf, parameters):
        cqueue, cbuffer = self.pm.getqueue()

        self.pm.runThread(cbuffer=cbuffer, func=self.fn, buffers=parameters,
                          threads=({"width": self.resolution, "height": self.resolution, "depth": 1}))

        self.pm.enqueue_blit(cbuffer, pixelbuf)

        self.pm.start_process(cbuffer)
        self.pm.wait_process(cbuffer)


if __name__ == '__main__':
    sh = SphericalHarmonicsMetal(resolution=256 * 4, code=12344112, color_map=3)
    coords, normals, textures, colors = sh.get_vertex()

    print(coords[0], normals[0], textures[0], colors[0], sep='\n\n')

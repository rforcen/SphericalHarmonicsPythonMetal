'''
pymetal extender->implements 2d kernel call
and buffer conversions
'''

from runmetal import PyMetal
from runmetal import Metal
import objc
import numpy as np


class py_metalMT(PyMetal):
    def __init__(self):
        super(PyMetal).__init__()

    def runThread(self, cbuffer, func, buffers, threads=None, label=None):
        desc = Metal.MTLComputePipelineDescriptor.new()
        if label is not None:
            desc.setLabel_(label)
        desc.setComputeFunction_(func)
        state = self.dev.newComputePipelineStateWithDescriptor_error_(
            desc, objc.NULL)
        encoder = cbuffer.computeCommandEncoder()
        encoder.setComputePipelineState_(state)
        bufmax = 0
        for i, buf in enumerate(buffers):
            encoder.setBuffer_offset_atIndex_(buf, 0, i)
            if bufmax < buf.length():
                bufmax = buf.length()

        # threads

        # number of thread per group
        w = state.threadExecutionWidth()
        h = max(1, int(state.maxTotalThreadsPerThreadgroup() / w))
        tpg = self.getmtlsize({"width": w, "height": h, "depth": 1})

        # number of thread per grig
        ntg = self.getmtlsize(threads)

        encoder.dispatchThreads_threadsPerThreadgroup_(ntg, tpg)
        encoder.endEncoding()

    def intBuffer(self, i):
        return self.numpybuffer(np.array(i, dtype=np.int32))

    def floatBuffer(self, f):
        return self.numpybuffer(np.array(f, dtype=np.float32))

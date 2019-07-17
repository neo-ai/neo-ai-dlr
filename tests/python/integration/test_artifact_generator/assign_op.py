"""Create artifact used in test_assign_op"""

import sys
import os

if len(sys.argv) != 2:
  print(f'Usage: {sys.argv[0]} [destination directory]')
  sys.exit(1)

import tvm
import nnvm.compiler
import nnvm.symbol as sym

w = sym.Variable('w')
w2 = sym.Variable('w2')
w = sym._assign(w, w + 1)
w2 = sym._assign(w2, w + 1)

dshape = (5, 3, 18, 18)
shape_dict = {'w':dshape, 'w2':dshape}
graph, lib, _ = nnvm.compiler.build(w2, tvm.target.create('llvm'), shape_dict)
params = {}

with open(os.path.join(sys.argv[1], 'compiled.json'), 'w') as f:
  f.write(graph.json())
lib.export_library(os.path.join(sys.argv[1], 'compiled.so'), cc='g++')
with open(os.path.join(sys.argv[1], 'compiled.params'), 'wb') as f:
  f.write(nnvm.compiler.save_param_dict(params))

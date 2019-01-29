# DLR

## Installation

### Building the share library
    mkdir build; cd build
    cmake ..; make -j4; cd ..
    
### Installing python binding
    cd python; python setup.py install; cd ..

### Verifying installation (on linux)
    cd tests/python/integration/
    python load_and_run_tvm_model.py
    python load_and_run_treelite_model.py

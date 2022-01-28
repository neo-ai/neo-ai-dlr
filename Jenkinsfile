#!/usr/bin/groovy
// -*- mode: groovy -*-
// Jenkins pipeline
// See documents at https://jenkins.io/doc/book/pipeline/jenkinsfile/

// List of cloud targets
def cloudTargetMatrix = [
  "c4", "c5", "m4", "m5"
]

def inferenceContainerApps = [
  ["xgboost", "cpu"], ["image_classification", "cpu"], ["image_classification", "gpu"]
]

/* Pipeline definition */
pipeline {
  // Each stage specify its own agent
  agent none

  // Set up common job properties
  options {
    ansiColor('xterm')                              // Use color in terminal
    timestamps()                                    // Show timestamp
    timeout(time: 120, unit: 'MINUTES')             // Timeout after 2 hours
    buildDiscarder(logRotator(numToKeepStr: '10'))  // Rotate build logs
  }

  // Build stages
  stages {
    stage('Jenkins: Get sources') {
      agent {
        label 'cpu-build'
      }
      steps {
        nodeInfo()
        checkoutSrcs()
        stash name: 'srcs', excludes: '.git/'
        milestone label: 'Sources ready', ordinal: 1
      }
    }
    stage('Lint') {
      agent {
        dockerfile {
          filename 'Dockerfile.cpu_bare'
          dir 'tests/ci_build'
          label 'ubuntu && amd64 && cpu-build'
          args '-v ${PWD}:/workspace -w /workspace'
        }
      }
      steps {
        nodeInfo()
        unstash name: 'srcs'
        sh """
        tests/ci_build/git-clang-format.sh HEAD~1
        tests/ci_build/git-clang-format.sh origin/$CHANGE_TARGET
        """
      }
    }
    stage('Build & Test') {
      parallel {
        stage('Build for Manylinux') {
          agent {
            dockerfile {
              filename 'Dockerfile.manylinux'
              dir 'tests/ci_build'
              label 'ubuntu && amd64 && cpu-build'
              args '-v ${PWD}:/workspace -w /workspace'
            }
          }
          steps {
            nodeInfo()
            unstash name: 'srcs'
            sh """
            mkdir -p build
            cd build
            cmake .. && make -j16
            cd ..
            tests/ci_build/create_wheel.sh manylinux1_x86_64
            """
            stash name: 'dlr_cpu_whl', includes: 'python/dist/*.whl'
          }
        }
        stage('Build for CPU') {
          agent {
            dockerfile {
              filename 'Dockerfile.cpu_bare'
              dir 'tests/ci_build'
              label 'ubuntu && amd64 && cpu-build'
              args '-v ${PWD}:/workspace -w /workspace'
            }
          }
          steps {
            nodeInfo()
            unstash name: 'srcs'
            sh """
            mkdir -p build
            cd build
            cmake -DENABLE_DATATRANSFORM=ON .. && make -j16
            CTEST_OUTPUT_ON_FAILURE=TRUE make test
            cd ..
            tests/ci_build/create_wheel.sh manylinux1_x86_64
            """
          }
        }
        stage('Build for Hexagon') {
          agent {
            dockerfile {
              filename 'Dockerfile.cpu_bare'
              dir 'tests/ci_build'
              label 'ubuntu && amd64 && cpu-build'
              args '-v ${PWD}:/workspace -w /workspace'
            }
          }
          steps {
            nodeInfo()
            unstash name: 'srcs'
            sh """
            mkdir -p build
            cd build
            cmake .. -DWITH_HEXAGON=1 && make -j16
            cd ..
            tests/ci_build/create_wheel.sh manylinux1_x86_64
            """
          }
        }
        stage('Build for GPU') {
          agent {
            dockerfile {
              filename 'Dockerfile.gpu_bare'
              dir 'tests/ci_build'
              label 'ubuntu && amd64 && gpu-build'
              args '-v ${PWD}:/workspace -w /workspace'
            }
          }
          steps {
            nodeInfo()
            unstash name: 'srcs'
            echo "Building artifact for AMD64 with GPU capability. Using CUDA 10.2, CuDNN 8, TensorRT 7.1"
            s3Download(file: 'tests/ci_build/TensorRT-7.1.3.4.Ubuntu-18.04.x86_64-gnu.cuda-10.2.cudnn8.0.tar.gz',
                      bucket: 'neo-ai-dlr-jenkins-artifacts',
                      path: 'TensorRT-7.1.3.4.Ubuntu-18.04.x86_64-gnu.cuda-10.2.cudnn8.0.tar.gz',
                      force:true)
            sh """
            tar xvzf tests/ci_build/TensorRT-7.1.3.4.Ubuntu-18.04.x86_64-gnu.cuda-10.2.cudnn8.0.tar.gz -C /workspace/
            mkdir -p build
            cd build
            cmake .. -DUSE_CUDA=ON -DUSE_CUDNN=ON -DUSE_TENSORRT=/workspace/TensorRT-7.1.3.4/ && make -j4
            cd ..
            tests/ci_build/create_wheel.sh ubuntu1804_cuda102_cudnn8_tensorrt71_x86_64
            """
          }
        }
      }
    }
    stage('Jenkins: Install & Test') {
      steps {
        script {
          parallel (cloudTargetMatrix.collectEntries{
            [(it): { CloudInstallAndTest(it) } ]
          })
        }
      }
    }
    stage('Jenkins: Build Container') {
      agent {
        label 'cpu-build'
      }
      steps {
        script {
          parallel (inferenceContainerApps.collectEntries{
            [(it[0] + '-' + it[1]): { BuildInferenceContainer(it[0], it[1]) } ]
          })
        }
      }
    }
  }
}

/* Function definitions to follow */

// Add more info about job node
def nodeInfo() {
  sh """
     echo "INFO: NODE_NAME=${NODE_NAME} EXECUTOR_NUMBER=${EXECUTOR_NUMBER}"
     """
}

// Check out source code
def checkoutSrcs() {
  retry(5) {
    try {
      timeout(time: 2, unit: 'MINUTES') {
        checkout scm
        sh 'git submodule update --init --recursive'
      }
    } catch (exc) {
      deleteDir()
      error "Failed to fetch source codes"
    }
  }
}
// Install and test DLR for cloud targets
def CloudInstallAndTest(cloudTarget) {
  def nodeReq = "ubuntu && amd64 && ${cloudTarget}"
  node(nodeReq) {
    sh """
    echo "INFO: NODE_NAME=${NODE_NAME} EXECUTOR_NUMBER=${EXECUTOR_NUMBER}"
    """
    echo "Installing DLR package for ${cloudTarget} target"
    if (cloudTarget == "p2" || cloudTarget == "p3") {
      unstash 'dlr_gpu_whl'
    } else {
      unstash 'dlr_cpu_whl'
    }
    sh """
    ls -lh python/dist/*.whl
    echo "Updating pip3..."
    sudo -H pip3 install -U pip setuptools wheel
    pip3 --version
    echo "Installing DLR Python package..."
    pip3 install python/dist/*.whl
    """
    echo "Running integration tests..."
    unstash name: 'srcs'
    sh """
    python3 tests/python/integration/load_and_run_tvm_model.py
    python3 tests/python/integration/load_and_run_treelite_model.py
    python3 -m pytest -v --fulltrace -s tests/python/unittest/test_get_set_input.py
    """
  }
}

// Build DLR inference containers
def BuildInferenceContainer(app, target) {
  def nodeReq = "ubuntu && amd64 && cpu-build"
  node(nodeReq) {
    sh """
    echo "INFO: NODE_NAME=${NODE_NAME} EXECUTOR_NUMBER=${EXECUTOR_NUMBER}"
    """
    unstash name: 'srcs'
    echo "Building inference container ${app} for target ${target}"
    if (target == "gpu") {
      // Download TensorRT library
      s3Download(file: 'container/TensorRT-7.1.3.4.Ubuntu-18.04.x86_64-gnu.cuda-10.2.cudnn8.0.tar.gz',
                 bucket: 'neo-ai-dlr-jenkins-artifacts',
                 path: 'TensorRT-7.1.3.4.Ubuntu-18.04.x86_64-gnu.cuda-10.2.cudnn8.0.tar.gz')
    }
    sh """
    docker build --build-arg APP=${app} -t ${app}-${target} -f container/Dockerfile.${target} .
    """
  }
}

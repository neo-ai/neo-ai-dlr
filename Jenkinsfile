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
        checkoutSrcs()
        stash name: 'srcs', excludes: '.git/'
        milestone label: 'Sources ready', ordinal: 1
      }
    }
    stage('Jenkins: Build') {
      steps {
        script {
          parallel ([ "build-amd64-cpu" : { AMD64BuildCPU() },
                      "build-amd64-gpu" : { AMD64BuildGPU() } ])
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

// Build for AMD64 CPU target
def AMD64BuildCPU() {
  def nodeReq = "ubuntu && amd64 && cpu-build"
  def dockerTarget = "cpu_bare"
  def dockerArgs = ""
  node(nodeReq) {
    unstash name: 'srcs'
    echo "Building universal artifact for AMD64, CPU-only"
    sh """
    tests/ci_build/ci_build.sh ${dockerTarget} ${dockerArgs} tests/ci_build/build_via_cmake.sh
    tests/ci_build/ci_build.sh ${dockerTarget} ${dockerArgs} tests/ci_build/create_wheel.sh manylinux1_x86_64
    """
    stash name: 'dlr_cpu_whl', includes: 'python/dist/*.whl'
  }
}

// Build for AMD64 + CUDA target
def AMD64BuildGPU() {
  def nodeReq = "ubuntu && amd64 && gpu-build"
  node(nodeReq) {
    unstash name: 'srcs'
    echo "Building artifact for AMD64 with GPU capability. Using CUDA 8.0, CuDNN 7, TensorRT 4"
    sh """
    tests/ci_build/build_via_cmake.sh -DUSE_CUDA=ON -DUSE_CUDNN=ON -DUSE_TENSORRT=/usr/src/tensorrt
    PYTHON_COMMAND=/opt/python/bin/python tests/ci_build/create_wheel.sh ubuntu1404_cuda80_cudnn7_tensorrt4_amd64
    """
    stash name: 'dlr_gpu_whl', includes: 'python/dist/*.whl'
  }
}

// Install and test DLR for cloud targets
def CloudInstallAndTest(cloudTarget) {
  def nodeReq = "ubuntu && amd64 && ${cloudTarget}"
  node(nodeReq) {
    echo "Installing DLR package for ${cloudTarget} target"
    if (cloudTarget == "p2" || cloudTarget == "p3") {
      unstash 'dlr_gpu_whl'
    } else {
      unstash 'dlr_cpu_whl'
    }
    sh """
    ls -lh python/dist/*.whl
    pip3 install python/dist/*.whl
    """
    if (cloudTarget == "p2" || cloudTarget == "p3") {
      sh """
      sudo pip3 install --upgrade --force-reinstall tensorflow_gpu
      """
    } else {
      sh """
      sudo pip3 install --upgrade --force-reinstall tensorflow
      """
    }
    sh """
    type toco_from_protos
    """
    echo "Running integration tests..."
    unstash name: 'srcs'
    sh """
    python3 tests/python/integration/load_and_run_tvm_model.py
    python3 tests/python/integration/load_and_run_treelite_model.py
    python3 -m pytest -v --fulltrace -s tests/python/unittest/test_get_set_input.py
    python3 -m pytest -v --fulltrace -s tests/python/unittest/test_tf_model.py
    python3 -m pytest -v --fulltrace -s tests/python/unittest/test_tflite_model.py
    """
  }
}

// Build DLR inference containers
def BuildInferenceContainer(app, target) {
  def nodeReq = "ubuntu && amd64 && cpu-build"
  node(nodeReq) {
    unstash name: 'srcs'
    echo "Building inference container ${app} for target ${target}"
    if (target == "gpu") {
      // Download TensorRT library
      s3Download(file: 'container/TensorRT-7.0.0.11.Ubuntu-18.04.x86_64-gnu.cuda-10.0.cudnn7.6.tar.gz',
                 bucket: 'neo-ai-dlr-jenkins-artifacts',
                 path: 'TensorRT-7.0.0.11.Ubuntu-18.04.x86_64-gnu.cuda-10.0.cudnn7.6.tar.gz')
    }
    sh """
    docker build --build-arg APP=${app} -t ${app}-${target} -f container/Dockerfile.${target} .
    """
  }
}

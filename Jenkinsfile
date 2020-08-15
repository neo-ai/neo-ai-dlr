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
    stage('Build & Test') {
      parallel {
        stage('Build & Test On CPU') {
          agent {
            dockerfile {
              filename 'Dockerfile.cpu_bare'
              dir 'tests/ci_build'
              label 'ubuntu && amd64 && cpu-build'
              args '-v ${PWD}:/workspace -w /workspace'
            }
          }
          steps {
            unstash name: 'srcs'
            sh """
            mkdir -p /workspace/dlr/lib/python/
            export PYTHONPATH=/workspace/dlr/lib/python/
            cd python
            python3 setup.py install --home=/workspace/dlr
            cd ..
            python3 tests/python/integration/load_and_run_tvm_model.py
            python3 tests/python/integration/load_and_run_treelite_model.py
            """
          }
        }
        stage('Build & Test On GPU') {
          agent {
            dockerfile {
              filename 'Dockerfile.gpu_bare'
              dir 'tests/ci_build'
              label 'ubuntu && amd64 && gpu-build'
              args '-v ${PWD}:/workspace -w /workspace --runtime nvidia'
            }
          }
          steps {
            unstash name: 'srcs'
            echo "Building artifact for AMD64 with GPU capability. Using CUDA 10.2, CuDNN 8, TensorRT 7.1"
            s3Download(file: 'tests/ci_build/TensorRT-7.0.0.11.Ubuntu-18.04.x86_64-gnu.cuda-10.0.cudnn7.6.tar.gz',
                      bucket: 'neo-ai-dlr-jenkins-artifacts',
                      path: 'TensorRT-7.0.0.11.Ubuntu-18.04.x86_64-gnu.cuda-10.0.cudnn7.6.tar.gz',
                      force:true)
            print(env.PATH)
            sh """
            python3 --version
            mkdir -p /workspace/dlr/lib/python/
            tar xvzf tests/ci_build/TensorRT-7.0.0.11.Ubuntu-18.04.x86_64-gnu.cuda-10.0.cudnn7.6.tar.gz -C /workspace/
            export PYTHONPATH=/workspace/dlr/lib/python/
            cd python
            python3 setup.py install --use-cuda --use-cudnn --use-tensorrt=/workspace/TensorRT-7.0.0.11/ --home=/workspace/dlr
            cd ..
            python3 tests/python/integration/load_and_run_tvm_model.py
            python3 tests/python/integration/load_and_run_treelite_model.py
            """
          }
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

#!/usr/bin/groovy
// -*- mode: groovy -*-
// Jenkins pipeline
// See documents at https://jenkins.io/doc/book/pipeline/jenkinsfile/

// List of cloud targets
def cloudTargetMatrix = [
  "c4", "c5", "m4", "m5"
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
    withAWS(credentials:'Neo-AI-CI-Fleet') {
      s3Upload bucket: 'neo-ai-dlr-jenkins-artifacts', acl: 'Private', path: "${env.JOB_NAME}/${env.BUILD_ID}/artifacts/", includePathPattern:'python/dist/**'
    }
  }
}

def AMD64BuildGPU() {
  def nodeReq = "ubuntu && amd64 && gpu-build"
  node(nodeReq) {
    unstash name: 'srcs'
    echo "Building artifact for AMD64 with GPU capability. Using CUDA 8.0, CuDNN 7, TensorRT 4"
    sh """
    tests/ci_build/build_via_cmake.sh -DUSE_CUDA=ON -DUSE_CUDNN=ON -DUSE_TENSORRT=/usr/src/tensorrt
    PYTHON_COMMAND=/opt/python/bin/python tests/ci_build/create_wheel.sh ubuntu1404_cuda80_cudnn7_tensorrt4_amd64
    """
    withAWS(credentials:'Neo-AI-CI-Fleet') {
      s3Upload bucket: 'neo-ai-dlr-jenkins-artifacts', acl: 'Private', path: "${env.JOB_NAME}/${env.BUILD_ID}/artifacts/", includePathPattern:'python/dist/**'
    }
  }
}

def CloudInstallAndTest(cloudTarget) {
  def nodeReq = "ubuntu && amd64 && ${cloudTarget}"
  node(nodeReq) {
    echo "Installing DLR package for ${cloudTarget} target"
    withAWS(credentials:'Neo-AI-CI-Fleet') {
      files = s3FindFiles(bucket: 'neo-ai-dlr-jenkins-artifacts', glob: "${env.JOB_NAME}/${env.BUILD_ID}/artifacts/dlr-*-manylinux1_x86_64.whl")
      assert files.size() == 1
      files.each {
        s3Download file: it.name, bucket: 'neo-ai-dlr-jenkins-artifacts', path: it.path, force: true
      }
    }
    sh """
    ls -lh *.whl
    pip3 install *.whl
    """
    echo "Running integration tests..."
    unstash name: 'srcs'
    sh """
    python3 -m pytest -v -s --fulltrace tests/python/integration/load_and_run_*.py
    """
  }
}

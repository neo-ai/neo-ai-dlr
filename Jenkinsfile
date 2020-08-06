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
      agent {
        dockerfile {
          filename 'Dockerfile.cpu_bare'
          dir 'tests/ci_build'
          label 'cpu-build'
          args '-v ${PWD}:/workspace -w /workspace'
        }
      }
      steps {
        sh """
        cd python
        python3 setup.py install --home=/workspace/dlr
        cd ..
        python3 tests/python/integration/load_and_run_tvm_model.py
        python3 tests/python/integration/load_and_run_treelite_model.py
        python3 -m pytest -v --fulltrace -s tests/python/unittest/test_get_set_input.py
        """
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

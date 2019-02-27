#!/usr/bin/groovy
// -*- mode: groovy -*-
// Jenkins pipeline
// See documents at https://jenkins.io/doc/book/pipeline/jenkinsfile/

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
        label 'cpu-bare'
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
          parallel ([ "build-amd64-cpu" : { AMD64BuildCPU() } ])
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
  def nodeReq = "ubuntu && amd64 && cpu-bare"
  def dockerTarget = "cpu_bare"
  def dockerArgs = ""
  node(nodeReq) {
    unstash name: 'srcs'
    echo "Building univeral artifact for AMD64, CPU-only"
    sh """
    tests/ci_build/ci_build.sh ${dockerTarget} ${dockerArgs} tests/ci_build/build_via_cmake.sh
    tests/ci_build/ci_build.sh ${dockerTarget} ${dockerArgs} tests/ci_build/create_wheel.sh
    """
    withAWS(credentials:'Neo-AI-CI-Fleet') {
      s3Upload bucket: 'neo-ai-dlr-jenkins-artifacts', acl: 'Private', path: "${env.JOB_NAME}/${env.BUILD_ID}/artifacts/", includePathPattern:'python/dist/**'
    }
  }
}

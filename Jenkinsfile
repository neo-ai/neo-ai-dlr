#!/usr/bin/groovy
// -*- mode: groovy -*-
// Jenkins pipeline
// See documents at https://jenkins.io/doc/book/pipeline/jenkinsfile/

/* Unrestricted tasks: tasks that do NOT generate artifacts.
 * Only use nodes labelled with "unrestricted" */

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
        label 'unrestricted'
      }
      steps {
        checkoutSrcs()
        stash name: 'srcs', excludes: '.git/'
        milestone label: 'Sources ready', ordinal: 1
      }
    }
    stage('Jenkins: Build & Test') {
      steps {
        script {
          parallel ([ "hello-world" : { buildHelloWorldJob() } ])
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
        sh 'git submodule update --init'
      }
    } catch (exc) {
      deleteDir()
      error "Failed to fetch source codes"
    }
  }
}

// Placeholder for a build task
def buildHelloWorldJob() {
  // Use Linux CPU worker
  def nodeReq = "linux && cpu && unrestricted"
  node(nodeReq) {
    unstash name: 'srcs'
    echo "Hello world!"
    sh """
    git rev-parse HEAD
    cmake --version
    g++ --version
    python3 --version
    apt-get moo
    """
  }
}

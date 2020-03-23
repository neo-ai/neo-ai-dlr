/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

package com.amazon.dlr.examples.classification;

import android.app.Activity;
import android.graphics.Bitmap;
import android.graphics.RectF;
import android.os.SystemClock;
import android.os.Trace;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.PriorityQueue;

import com.amazon.dlr.examples.classification.dlr.ClassifierPass;
import com.amazon.dlr.examples.classification.dlr.DLRGluonCVMobileNetV2_075;
import com.amazon.dlr.examples.classification.dlr.DLRGluonCVMobileNetV2_100;
import com.amazon.dlr.examples.classification.dlr.DLRGluonCVResNet18;
import com.amazon.dlr.examples.classification.dlr.DLRGluonCVResNet50;
import com.amazon.dlr.examples.classification.dlr.DLRKerasMobileNetV2;
import com.amazon.dlr.examples.classification.dlr.DLRTensorFlowMobileNetV1;
import com.amazon.dlr.examples.classification.env.Logger;

/** A classifier specialized to label images using TensorFlow Lite. */
public abstract class Classifier {
  private static final Logger LOGGER = new Logger();

  /** The model type used for classification. */
  public enum Model {
    TENSORFLOW_MOBILENET_V1,
    KERAS_MOBILENET_V2,
    GLUONCV_MOBILENET_V2_075,
    GLUONCV_MOBILENET_V2_100,
    GLUONCV_RESNET18_V2,
    GLUONCV_RESNET50_V2,
    PASS
  }

  /** The runtime device type used for executing classification. */
  public enum Device {
    CPU,
  }

  /** Number of results to show in the UI. */
  private static final int MAX_RESULTS = 3;

  /** Dimensions of inputs. */
  private static final int DIM_BATCH_SIZE = 1;

  private static final int DIM_PIXEL_SIZE = 3;

  /** Preallocated buffers for storing image data in. */
  private final int[] intValues = new int[getImageSizeX() * getImageSizeY()];

  /** Labels corresponding to the output of the vision model. */
  private List<String> labels;

  /** A ByteBuffer to hold image data, to be feed into Tensorflow Lite as inputs. */
  protected ByteBuffer imgData = null;

  /**
   * Creates a classifier with the provided configuration.
   *
   * @param activity The current Activity.
   * @param model The model to use for classification.
   * @param device The device to use for classification.
   * @param numThreads The number of threads to use for classification.
   * @return A classifier with the desired configuration.
   */
  public static Classifier create(Activity activity, Model model, Device device, int numThreads)
      throws IOException {
    if (model == Model.TENSORFLOW_MOBILENET_V1) {
      return new DLRTensorFlowMobileNetV1(activity);
    }
    if (model == Model.KERAS_MOBILENET_V2) {
      return new DLRKerasMobileNetV2(activity);
    }
    if (model == Model.GLUONCV_MOBILENET_V2_075) {
      return new DLRGluonCVMobileNetV2_075(activity);
    }
    if (model == Model.GLUONCV_MOBILENET_V2_100) {
      return new DLRGluonCVMobileNetV2_100(activity);
    }
    if (model == Model.GLUONCV_RESNET18_V2) {
      return new DLRGluonCVResNet18(activity);
    }
    if (model == Model.GLUONCV_RESNET50_V2) {
      return new DLRGluonCVResNet50(activity);
    }
    return new ClassifierPass(activity, device, numThreads);
  }

  /** An immutable result returned by a Classifier describing what was recognized. */
  public static class Recognition {
    /**
     * A unique identifier for what has been recognized. Specific to the class, not the instance of
     * the object.
     */
    private final String id;

    /** Display name for the recognition. */
    private final String title;

    /**
     * A sortable score for how good the recognition is relative to others. Higher should be better.
     */
    private final Float confidence;

    /** Optional location within the source image for the location of the recognized object. */
    private RectF location;

    public Recognition(
        final String id, final String title, final Float confidence, final RectF location) {
      this.id = id;
      this.title = title;
      this.confidence = confidence;
      this.location = location;
    }

    public String getId() {
      return id;
    }

    public String getTitle() {
      return title;
    }

    public Float getConfidence() {
      return confidence;
    }

    public RectF getLocation() {
      return new RectF(location);
    }

    public void setLocation(RectF location) {
      this.location = location;
    }

    @Override
    public String toString() {
      String resultString = "";
      if (id != null) {
        resultString += "[" + id + "] ";
      }

      if (title != null) {
        resultString += title + " ";
      }

      if (confidence != null) {
        resultString += String.format("(%.1f%%) ", confidence * 100.0f);
      }

      if (location != null) {
        resultString += location + " ";
      }

      return resultString.trim();
    }
  }

  protected Classifier(Activity activity) throws IOException {
    labels = loadLabelList(activity);
    imgData =
            ByteBuffer.allocateDirect(
                    DIM_BATCH_SIZE
                            * getImageSizeX()
                            * getImageSizeY()
                            * DIM_PIXEL_SIZE
                            * getNumBytesPerChannel());
    imgData.order(ByteOrder.nativeOrder());
    LOGGER.d("Created a Tensorflow Lite Image Classifier.");
  }

  /** Initializes a {@code Classifier}. */
  protected Classifier(Activity activity, Device device, int numThreads) throws IOException {
    labels = loadLabelList(activity);
    imgData =
        ByteBuffer.allocateDirect(
            DIM_BATCH_SIZE
                * getImageSizeX()
                * getImageSizeY()
                * DIM_PIXEL_SIZE
                * getNumBytesPerChannel());
    imgData.order(ByteOrder.nativeOrder());
    LOGGER.d("Created a Tensorflow Lite Image Classifier.");
  }

  /** Reads label list from Assets. */
  private List<String> loadLabelList(Activity activity) throws IOException {
    List<String> labels = new ArrayList<String>();
    BufferedReader reader =
        new BufferedReader(new InputStreamReader(activity.getAssets().open(getLabelPath())));
    String line;
    while ((line = reader.readLine()) != null) {
      labels.add(line);
    }
    reader.close();
    return labels;
  }

  /** Writes Image data into a {@code ByteBuffer}. */
  private void convertBitmapToByteBuffer(Bitmap bitmap) {
    if (imgData == null) {
      return;
    }
    imgData.rewind();
    bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
    // Convert the image to floating point.
    int pixel = 0;
    long startTime = SystemClock.uptimeMillis();
    for (int i = 0; i < getImageSizeX(); ++i) {
      for (int j = 0; j < getImageSizeY(); ++j) {
        final int val = intValues[pixel++];
        addPixelValue(val);
      }
    }
    long endTime = SystemClock.uptimeMillis();
    LOGGER.v("Timecost to put values into ByteBuffer: " + (endTime - startTime));
  }

  /** Runs inference and returns the classification results. */
  public List<Recognition> recognizeImage(final Bitmap bitmap) {
    // Log this method so that it can be analyzed with systrace.
    Trace.beginSection("recognizeImage");

    Trace.beginSection("preprocessBitmap");
    convertBitmapToByteBuffer(bitmap);
    Trace.endSection();

    // Run the inference call.
    Trace.beginSection("runInference");
    long startTime = SystemClock.uptimeMillis();
    runInference();
    long endTime = SystemClock.uptimeMillis();
    Trace.endSection();
    LOGGER.v("Timecost to run model inference: " + (endTime - startTime));

    // Find the best classifications.
    PriorityQueue<Recognition> pq =
        new PriorityQueue<Recognition>(
            3,
            new Comparator<Recognition>() {
              @Override
              public int compare(Recognition lhs, Recognition rhs) {
                // Intentionally reversed to put high confidence at the head of the queue.
                return Float.compare(rhs.getConfidence(), lhs.getConfidence());
              }
            });
    for (int i = 0; i < labels.size(); ++i) {
      pq.add(
          new Recognition(
              "" + i,
              labels.size() > i ? labels.get(i) : "unknown",
              getNormalizedProbability(i),
              null));
    }
    final ArrayList<Recognition> recognitions = new ArrayList<Recognition>();
    int recognitionsSize = Math.min(pq.size(), MAX_RESULTS);
    for (int i = 0; i < recognitionsSize; ++i) {
      recognitions.add(pq.poll());
    }
    Trace.endSection();
    return recognitions;
  }

  /** Closes the interpreter and model to release resources. */
  public void close() {
  }

  /**
   * Get the image size along the x axis.
   *
   * @return
   */
  public abstract int getImageSizeX();

  /**
   * Get the image size along the y axis.
   *
   * @return
   */
  public abstract int getImageSizeY();

  /**
   * Get the name of the model file stored in Assets.
   *
   * @return
   */
  protected abstract String getModelPath();

  /**
   * Get the name of the label file stored in Assets.
   *
   * @return
   */
  protected abstract String getLabelPath();

  /**
   * Get the number of bytes that is used to store a single color channel value.
   *
   * @return
   */
  protected abstract int getNumBytesPerChannel();

  /**
   * Add pixelValue to byteBuffer.
   *
   * @param pixelValue
   */
  protected abstract void addPixelValue(int pixelValue);

  /**
   * Read the probability value for the specified label This is either the original value as it was
   * read from the net's output or the updated value after the filter was applied.
   *
   * @param labelIndex
   * @return
   */
  protected abstract float getProbability(int labelIndex);

  /**
   * Set the probability value for the specified label.
   *
   * @param labelIndex
   * @param value
   */
  protected abstract void setProbability(int labelIndex, Number value);

  /**
   * Get the normalized probability value for the specified label. This is the final value as it
   * will be shown to the user.
   *
   * @return
   */
  protected abstract float getNormalizedProbability(int labelIndex);

  /**
   * Run inference using the prepared input in {@link #imgData}. Afterwards, the result will be
   * provided by getProbability().
   *
   * <p>This additional method is necessary, because we don't have a common base for different
   * primitive data types.
   */
  protected abstract void runInference();

  /**
   * Get the total number of labels.
   *
   * @return
   */
  protected int getNumLabels() {
    return labels.size();
  }
}

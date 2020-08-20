# coding: utf-8
from __future__ import print_function
from dlr import DLRModel
import numpy as np
import shutil
import tensorflow as tf
import tensorflow.lite as lite
import uuid
# https://github.com/tensorflow/tensorflow/issues/18165
tf.compat.v1.disable_eager_execution()

TFLITE_FILE_PATH = "/tmp/test_model.tflite"


def _generate_tflite_file():
    tf_saved_model_path = "/tmp/test-tf-model-{}".format(str(uuid.uuid1()))
    a = tf.compat.v1.placeholder(tf.float32, shape=[2, 2], name="input1")
    b = tf.compat.v1.placeholder(tf.float32, shape=[2, 2], name="input2")
    ab = tf.matmul(a, b)
    mm = tf.matmul(a, ab, name="preproc/mm")
    out0 = tf.square(mm, name="preproc/output1")
    mm_flat = tf.reshape(mm, shape=[-1])
    out1 = tf.argmax(mm_flat, name="preproc/output2")
    try:
        with tf.compat.v1.Session() as sess:
            tf.compat.v1.saved_model.simple_save(sess, tf_saved_model_path,
                                                 inputs={"input1": a, "input2": b},
                                                 outputs={"preproc/output1": out0,
                                                          "preproc/output2": out1})
        converter = lite.TFLiteConverter.from_saved_model(tf_saved_model_path)
        tflite_model = converter.convert()
        with open(TFLITE_FILE_PATH, "wb") as f:
            f.write(tflite_model)
    finally:
        shutil.rmtree(tf_saved_model_path, ignore_errors=True)


def test_tflite_model():
    _generate_tflite_file()

    m = DLRModel(TFLITE_FILE_PATH)
    inp_names = m.get_input_names()
    assert sorted(inp_names) == ['input1', 'input2']

    out_names = m.get_output_names()
    assert sorted(out_names) == ['preproc/output1', 'preproc/output2']

    inp1 = np.array([[4., 1.], [3., 2.]]).astype("float32")
    inp2 = np.array([[0., 1.], [1., 0.]]).astype("float32")

    res = m.run({'input1': inp1, 'input2': inp2})
    assert res is not None
    assert len(res) == 2
    exp_out0 = np.array([[36., 361.], [49.,  324.]]).astype("float32")
    assert np.alltrue(res[0] == exp_out0)
    assert res[1] == 1

    m_inp1 = m.get_input('input1')
    m_inp2 = m.get_input('input2')
    assert np.alltrue(m_inp1 == inp1)
    assert np.alltrue(m_inp2 == inp2)


if __name__ == '__main__':
    test_tflite_model()
    print('All tests passed!')

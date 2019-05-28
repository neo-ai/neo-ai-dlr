# coding: utf-8
from __future__ import print_function
from dlr import DLRModel
import numpy as np

TFLITE_FILE_PATH = "/tmp/test_model.tflite"


def _generate_tflite_file():
    import tensorflow as tf
    import tensorflow.lite as lite
    a = tf.placeholder(tf.float32, shape=[2, 2], name="input1")
    b = tf.placeholder(tf.float32, shape=[2, 2], name="input2")
    ab = tf.matmul(a, b)
    mm = tf.matmul(a, ab, name="preproc/mm")
    out0 = tf.square(mm, name="preproc/output1")
    mm_flat = tf.reshape(mm, shape=[-1])
    out1 = tf.argmax(mm_flat, name="preproc/output2")
    with tf.Session() as sess:
        converter = lite.TFLiteConverter.from_session(sess, input_tensors=[a, b], output_tensors=[out0, out1])
        tflite_model = converter.convert()
        with open(TFLITE_FILE_PATH, "wb") as f:
            f.write(tflite_model)


def test_tflite_model():
    _generate_tflite_file()

    m = DLRModel(TFLITE_FILE_PATH)
    inp_names = m.get_input_names()
    assert inp_names == ['input1', 'input2']

    out_names = m.get_output_names()
    assert out_names == ['preproc/output1', 'preproc/output2']

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

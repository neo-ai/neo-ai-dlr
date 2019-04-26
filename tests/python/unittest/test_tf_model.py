# coding: utf-8
from dlr.tf_model import TFModelImpl
import numpy as np

FROZEN_GRAPH_PATH = "/tmp/test_graph.pb"


def _generate_frozen_graph():
    import tensorflow as tf
    a = tf.placeholder(tf.float32, shape=[2, 2], name="input1")
    b = tf.placeholder(tf.float32, shape=[2, 2], name="input2")
    ab = tf.matmul(a, b)
    mm = tf.matmul(a, ab)
    tf.square(mm, name="output1")
    mm_flat = tf.reshape(mm, shape=[-1])
    tf.argmax(mm_flat, name="output2")

    with tf.Session() as sess:
        output_graph_def = tf.compat.v1.graph_util.convert_variables_to_constants(
            sess,
            tf.get_default_graph().as_graph_def(),
            ["output1", "output2"]
        )

        with tf.gfile.GFile(FROZEN_GRAPH_PATH, "wb") as f:
            f.write(output_graph_def.SerializeToString())
    sess.close()


def test_tf_model(dev_type=None, dev_id=None):
    _generate_frozen_graph()
    m = TFModelImpl(FROZEN_GRAPH_PATH, dev_type, dev_id)

    inp_names = m.get_input_names()
    assert inp_names == ['import/input1:0', 'import/input2:0']

    out_names = m.get_output_names()
    assert out_names == ['import/output1:0', 'import/output2:0']

    inp1 = [[4., 1.], [3., 2.]]
    inp2 = [[0., 1.], [1., 0.]]

    res = m.run({'import/input1:0': inp1, 'import/input2:0': inp2})
    assert res is not None
    assert len(res) == 2
    assert np.alltrue(res[0] == [[36., 361.], [49.,  324.]])
    assert res[1] == 1

    m_inp1 = m.get_input('import/input1:0')
    m_inp2 = m.get_input('import/input2:0')
    assert np.alltrue(m_inp1 == inp1)
    assert np.alltrue(m_inp2 == inp2)


def test_tf_model_on_cpu_0():
    test_tf_model("cpu", 0)


def test_tf_model_on_gpu_0():
    test_tf_model("gpu", 0)

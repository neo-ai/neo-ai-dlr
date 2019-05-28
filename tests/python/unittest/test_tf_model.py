# coding: utf-8
from __future__ import print_function
from dlr import DLRModel
import numpy as np

FROZEN_GRAPH_PATH = "/tmp/test_graph.pb"


def _generate_frozen_graph():
    import tensorflow as tf
    graph = tf.get_default_graph()
    a = tf.placeholder(tf.float32, shape=[2, 2], name="input1")
    b = tf.placeholder(tf.float32, shape=[2, 2], name="input2")
    ab = tf.matmul(a, b)
    mm = tf.matmul(a, ab, name="preproc/mm")
    tf.square(mm, name="preproc/output1")
    id1 = tf.identity(b, "preproc/id1")
    with graph.control_dependencies([id1]):
        mm_flat = tf.reshape(mm, shape=[-1])
        tf.argmax(mm_flat, name="preproc/output2")
    with tf.Session() as sess:
        output_graph_def = tf.graph_util.convert_variables_to_constants(
            sess,
            graph.as_graph_def(),
            ["preproc/output1", "preproc/output2"]
        )
        with tf.gfile.GFile(FROZEN_GRAPH_PATH, "wb") as f:
            f.write(output_graph_def.SerializeToString())


def test_tf_model(dev_type=None, dev_id=None):
    _generate_frozen_graph()
    model = DLRModel(FROZEN_GRAPH_PATH, dev_type, dev_id)
    inp_names = model.get_input_names()
    assert inp_names == ['import/input1:0', 'import/input2:0']

    out_names = model.get_output_names()
    assert out_names == ['import/preproc/output1:0', 'import/preproc/output2:0']

    inp1 = [[4., 1.], [3., 2.]]
    inp2 = [[0., 1.], [1., 0.]]

    res = model.run({'import/input1:0': inp1, 'import/input2:0': inp2})
    assert res is not None
    assert len(res) == 2
    assert np.alltrue(res[0] == [[36., 361.], [49.,  324.]])
    assert res[1] == 1

    m_inp1 = model.get_input('import/input1:0')
    m_inp2 = model.get_input('import/input2:0')
    assert np.alltrue(m_inp1 == inp1)
    assert np.alltrue(m_inp2 == inp2)


def test_tf_model_on_cpu_0():
    test_tf_model("cpu", 0)


def test_tf_model_on_gpu_0():
    test_tf_model("gpu", 0)


if __name__ == '__main__':
    test_tf_model()
    print('All tests passed!')
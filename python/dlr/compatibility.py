# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""Tool to upgrade TensorRT serialized Relay json artifacts (.tensorrt)
from older versions."""

import json

def create_updater(node_map, from_ver, to_ver):
    """Create an updater to update json loaded data.

    Parameters
    ----------
    node_map : Map[str, Function]
        Map from type_key to updating function

    from_ver : str
        Prefix of version that we can accept,

    to_ver : str
        The target version.

    Returns
    -------
    fupdater : function
        The updater function
    """
    def _updater(data):
        nodes = data["nodes"]
        for idx, item in enumerate(nodes):
            f = node_map.get(item["type_key"], None)
            if f:
                nodes[idx] = f(item, nodes)
        data["attrs"]["tvm_version"] = to_ver
        return data
    return _updater


def create_updater_06_to_07():
    """Create an update to upgrade json from v0.6 to v0.7

    Returns
    -------
    fupdater : function
        The updater function
    """
    def _ftype_var(item, nodes):
        vindex = int(item["attrs"]["var"])
        item["attrs"]["name_hint"] = nodes[vindex]["attrs"]["name"]
        # set vindex to null
        nodes[vindex]["type_key"] = ""
        del item["attrs"]["var"]
        assert item["type_key"].startswith("relay.")
        item["type_key"] = item["type_key"][len("relay."):]
        return item

    def _rename(new_name):
        def _convert(item, _):
            item["type_key"] = new_name
            return item
        return _convert

    def _update_tir_var(new_name):
        def _convert(item, _):
            item["type_key"] = new_name
            item["attrs"]["type_annotation"] = "0"
            return item
        return _convert

    def _update_global_key(item, _):
        item["repr_str"] = item["global_key"]
        del item["global_key"]
        return item

    def _update_op(item, _):
        item["repr_str"] = item["global_key"]
        del item["global_key"]
        if item["repr_str"] == "contrib.adaptive_avg_pool2d":
            item["repr_str"] = "nn.adaptive_avg_pool2d"
        elif item["repr_str"] == "contrib.adaptive_max_pool2d":
            item["repr_str"] = "nn.adaptive_max_pool2d"
        return item

    def _update_resize_attrs(item, _):
        if "align_corners" in item["attrs"]:
            if item["attrs"]["align_corners"] == "1":
                item["attrs"]["coordinate_transformation_mode"] = "align_corners"
            else:
                item["attrs"]["coordinate_transformation_mode"] = "half_pixel"
            del item["attrs"]["align_corners"]
        return item

    node_map = {
        # Base IR
        "SourceName": _update_global_key,
        "EnvFunc": _update_global_key,
        "relay.Op": _update_op,
        "relay.TypeVar": _ftype_var,
        "relay.GlobalTypeVar": _ftype_var,
        "relay.Type": _rename("Type"),
        "relay.TupleType": _rename("TupleType"),
        "relay.TypeConstraint": _rename("TypeConstraint"),
        "relay.FuncType": _rename("FuncType"),
        "relay.IncompleteType": _rename("IncompleteType"),
        "relay.TypeRelation": _rename("TypeRelation"),
        "relay.TypeCall": _rename("TypeCall"),
        "relay.Module": _rename("IRModule"),
        "relay.SourceName": _rename("SourceName"),
        "relay.Span": _rename("Span"),
        "relay.GlobalVar": _rename("GlobalVar"),
        "relay.Pass": _rename("transform.Pass"),
        "relay.PassInfo": _rename("transform.PassInfo"),
        "relay.PassContext": _rename("transform.PassContext"),
        "relay.ModulePass": _rename("transform.ModulePass"),
        "relay.Sequential": _rename("transform.Sequential"),
        "relay.attrs.ResizeAttrs": _update_resize_attrs,
        # TIR
        "Variable": _update_tir_var("tir.Var"),
        "SizeVar": _update_tir_var("tir.SizeVar"),
    }
    return create_updater(node_map, "0.6", "0.7")

def check_tensorrt_compatibility(file_path):
    # Check if conversion is needed.
    with open(file_path, 'r') as f:
        graph = json.load(f)
        if "subgraphs" in graph:
            return
    print("Model .tensorrt artifact was created from an earlier version of Neo "
          "which is not compatible with this DLR. Performing one-time conversion "
          "to new artifact format. If errors occur, please recompile your model.")
    with open(file_path, 'r') as f:
        graph = json.load(f)
    graph = create_updater_06_to_07()(graph)

    # Get body (call)
    body_node = graph["nodes"][int(graph["root"])]
    # Find all var nodes
    var_nodes = []
    for i, node in enumerate(graph["nodes"]):
        if node["type_key"] == "relay.Var":
            var_nodes.append([i, node])
    var_nodes = sorted(var_nodes, key=lambda x: graph["nodes"][int(x[1]["attrs"]["vid"])]["attrs"]["name_hint"])

    def add_node(graph, node):
        graph["nodes"].append(node)
        return str(len(graph["nodes"]) - 1)

    # Wrap body in Function() 
    arg_types = {'type_key': 'Array', 'data': [int(x[1]["attrs"]["_checked_type_"]) for x in var_nodes]}
    func_type = {
        'type_key': 'FuncType',
        'attrs': {
            'arg_types': add_node(graph, arg_types),
            'ret_type': body_node["attrs"]["_checked_type_"],
            'span': '0',
            'type_constraints': add_node(graph, {'type_key': 'Array'}),
            'type_params': add_node(graph, {'type_key': 'Array'}),
        }
    }
    func_attrs_dict = {
        'type_key': 'StrMap',
        'keys': ['Inline', 'Compiler', 'global_symbol', 'Primitive'],
        'data': [int(add_node(graph, {'type_key': 'IntImm', 'attrs': {'dtype': 'int32', 'value': '1'}})),
                 int(add_node(graph, {'type_key': 'runtime.String', 'repr_str': 'tensorrt'})),
                 int(add_node(graph, {'type_key': 'runtime.String', 'repr_str': 'tensorrt_0'})),
                 int(add_node(graph, {'type_key': 'IntImm', 'attrs': {'dtype': 'int32', 'value': '1'}}))]}
    func_attrs = {
        'type_key': 'DictAttrs',
        'attrs': {
            '__dict__': add_node(graph, func_attrs_dict)
        }
    }
    func_params = {'type_key': 'Array', 'data': [int(x[0]) for x in var_nodes]}  
    function_node = {
        'type_key': 'relay.Function',
        'attrs': {
            '_checked_type_': add_node(graph, func_type),
            'attrs': add_node(graph, func_attrs),
            'body': str(graph["root"]),
            'params': add_node(graph, func_params),
            'ret_type': body_node["attrs"]["_checked_type_"],
            'span': '0',
            'type_params': add_node(graph, {'type_key': 'Array'}),
        }
    }
    graph["root"] = int(add_node(graph, function_node))
    # Make high level map of subgraph name to serialized subgraph
    new_graph = {}
    new_graph["subgraphs"] = {"tensorrt_0": json.dumps(graph)}

    with open(file_path, 'w') as f:
        json.dump(new_graph, f)

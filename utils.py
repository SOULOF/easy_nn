import random
from nn import Placeholder


def forward_and_backward(outputnode, graph):
    # execute all the forward method of sorted_nodes.

    ## In practice, it's common to feed in mutiple data example in each forward pass rather than just 1. Because the examples can be processed in parallel. The number of examples is called batch size.
    for n in graph:
        n.forward()
        ## each node execute forward, get self.value based on the topological sort result.

    for n in graph[::-1]:
        n.backward()

    # return outputnode.value


###   v -->  a -->  C
##    b --> C
##    b --> v -- a --> C
##    v --> v ---> a -- > C

def toplogic(graph):
    sorted_node = []

    while len(graph) > 0:

        all_inputs = []
        all_outputs = []

        for n in graph:
            all_inputs += graph[n]
            all_outputs.append(n)

        all_inputs = set(all_inputs)
        all_outputs = set(all_outputs)

        need_remove = all_outputs - all_inputs  # which in all_inputs but not in all_outputs

        if len(need_remove) > 0:
            node = random.choice(list(need_remove))

            need_to_visited = [node]

            if len(graph) == 1: need_to_visited += graph[node]

            graph.pop(node)
            sorted_node += need_to_visited

            for _, links in graph.items():
                if node in links: links.remove(node)
        else:  # have cycle
            break

    return sorted_node


from collections import defaultdict


def convert_feed_dict_to_graph(feed_dict):
    computing_graph = defaultdict(list)

    nodes = [n for n in feed_dict]

    while nodes:
        n = nodes.pop(0)

        if isinstance(n, Placeholder):
            n.value = feed_dict[n]

        if n in computing_graph: continue

        for m in n.outputs:
            computing_graph[n].append(m)
            nodes.append(m)

    return computing_graph


def topological_sort_feed_dict(feed_dict):
    graph = convert_feed_dict_to_graph(feed_dict)

    return toplogic(graph)


def optimize(trainables, learning_rate=1e-2):
    # there are so many other update / optimization methods
    # such as Adam, Mom,
    for t in trainables:
        t.value += -1 * learning_rate * t.gradients[t]
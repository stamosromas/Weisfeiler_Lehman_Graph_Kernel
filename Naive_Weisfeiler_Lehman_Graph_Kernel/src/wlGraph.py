import networkx as nx
import pylab
from networkx.drawing.nx_pydot import graphviz_layout


class WLGraph:
    UNSEEN_LABEL = 0

    def __init__(self, T):
        self.T = T

    def computeMultiLabelString(self):
        multStrs = []
        for node in self.T.nodes():
            multStrs.append(self.computeStrOfNode(node))
        return sorted(list(set(multStrs)))

    def computeMaxLabel(self):
        return max([self.T.nodes[node]['label'] for node in self.T.nodes()])

    def computeStrOfNode(self, node):
        st = str(self.T.nodes[node]['label']) + '.'
        nodeNeigh = self.T.neighbors(node)
        nodeLabels = sorted([self.T.nodes[n]['label'] for n in nodeNeigh])
        for nodeLabel in nodeLabels:
            st += str(nodeLabel) + '.'
        st = st[:-1]
        return st

    def relabelingNodes(self, newLabelsMap):

        g = nx.Graph()
        for node in self.T.nodes():
            g.add_node(node, label=newLabelsMap[self.computeStrOfNode(node)])
        g.add_edges_from(self.T.edges())
        self.T = g

    def relabelingNodesTolerant(self, newLabelsMap):
        g = nx.Graph()
        for node in self.T.nodes():
            key = self.computeStrOfNode(node)
            g.add_node(node, label=newLabelsMap.get(key, self.UNSEEN_LABEL))
        g.add_edges_from(self.T.edges())
        self.T = g

    def getGraphLabels(self):
        return [self.T.nodes[node]['label'] for node in self.T.nodes()]

    def plotWithLabels(self):
        pos = graphviz_layout(self.T, prog="dot")
        labels = nx.get_node_attributes(self.T, 'label')
        nx.draw(self.T, pos, labels=labels, node_size=1000)
        pylab.show()

    def returnOutDegree(self, node):
        return self.T.out_degree(node)
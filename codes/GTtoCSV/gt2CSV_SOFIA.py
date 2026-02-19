"""
code Sofia

adpapted from Franca to read Renier data into CSV
"""
import os
import numpy
import random
from graph_tool.all import *
import sys
#value = sys.argv[1]


folder = "/home/admin/Ana/MicroBrain/CSV"

class ReadWriteGraph(object):
    def __init__(self, file = False):
        if file:
            self.graph = load_graph(file)
            print(list(self.graph.gp.keys()))
            print(list(self.graph.vp.keys()))
            print(list(self.graph.ep.keys()))
            return


    def loadGraphFromFile(self, file = False):
        self.graph = load_graph(file)

    def folderPrefix(self, path = folder):
        self.prefix = path

    def saveFile(self, array, filepath):
        if hasattr(self, 'prefix'):
            filepath = os.path.join(self.prefix, filepath)
        numpy.savetxt(filepath, array, delimiter=",")

    def writeEdges(self, file = folder +'edges.csv'):
        array = self.graph.get_edges()
        self.saveFile(array, file)

    def writeEdgePropertyRadii(self, file = folder +'radii_edge.csv'):
        array = self.graph.edge_properties.radii.get_array()
        self.saveFile(array, file)

    def writeEdgePropertyRadii_Atlas(self, file = folder +'radii_edge_atlas.csv'):
        array = self.graph.edge_properties.radii_atlas.get_array()
        self.saveFile(array, file)

    def writeEdgePropertyVein(self, file = folder +'vein.csv'):
        array = self.graph.edge_properties.vein.get_array()
        self.saveFile(array, file)

    def writeEdgePropertyArtery(self, file = folder +'artery.csv'):
        array = self.graph.edge_properties.artery_binary.get_array()
        self.saveFile(array, file)

    def writeEdgePropertyLength(self, file = folder +'length.csv'):
        array = self.graph.edge_properties.length.get_array()
        self.saveFile(array, file)

    def writeVertices(self, file = folder +'vertices.csv'):
        array = self.graph.get_vertices()
        self.saveFile(array, file)

    def writeVertexPropertyCoordinates(self, file = folder + 'coordinates.csv'):
        array = numpy.transpose(self.graph.vertex_properties.coordinates.get_2d_array((0,1,2)))
        self.saveFile(array, file)

    def writeVertexPropertyRadii(self, file = folder +'radii.csv'):
        array = self.graph.vertex_properties.radii.get_array()
        self.saveFile(array, file)

    def writeVertexPropertyCoordinatesAtlas(self, file =folder + 'coordinates_atlas.csv'):
        array = numpy.transpose(self.graph.vertex_properties.coordinates_atlas.get_2d_array((0,1,2)))
        self.saveFile(array, file)

    def writeVertexPropertyRadiiAtlas(self, file = folder +'radii_atlas.csv'):
        array = self.graph.vertex_properties.radii_atlas.get_array()
        self.saveFile(array, file)

    def writeVertexPropertyAnnotation(self, file = folder +'annotation.csv'):
        array = self.graph.vertex_properties.annotation.get_array()
        self.saveFile(array, file)

    def writeVertexPropertyDistanceToSurface(self, file = folder +'distance_to_surface.csv'):
        array = self.graph.vertex_properties.distance_to_surface.get_array()
        self.saveFile(array, file)

    def writeGraphPropertyEdgeGeometryIndices(self, file = folder +'edge_geometry_indices.csv'):
        array = numpy.transpose(self.graph.edge_properties.edge_geometry_indices.get_2d_array((0,1)))
        #array = self.graph.edge_properties.edge_geometry_indices.get_2d_array((0,1))
        self.saveFile(array, file)

    def writeGraphPropertyEdgeGeometryCoordinates(self, file = folder +'edge_geometry_coordinates.csv'):
        array = self.graph.graph_properties.edge_geometry_coordinates
        self.saveFile(array, file)

    def writeGraphPropertyEdgeGeometryRadii(self, file = folder +'edge_geometry_radii.csv'):
        array = self.graph.graph_properties.edge_geometry_radii
        self.saveFile(array, file)

    def writeGraphPropertyEdgeGeometryRadiiAtlas(self, file = folder +'edge_geometry_radii_atlas.csv'):
        array = self.graph.graph_properties.edge_geometry_radii_atlas
        self.saveFile(array, file)

    def writeGraphPropertyEdgeGeometryArteryBinary(self, file = folder +'edge_geometry_artery_binary.csv'):
        array = self.graph.graph_properties.edge_geometry_artery_binary
        self.saveFile(array, file)

    def writeGraphPropertyEdgeGeometryArteryRaw(self, file = folder +'edge_geometry_artery_raw.csv'):
        array = self.graph.graph_properties.edge_geometry_artery_raw
        self.saveFile(array, file)

    def writeGraphPropertyEdgeGeometryCoordinatesAtlas(self, file = folder +'edge_geometry_coordinates_atlas.csv'):
        array = self.graph.graph_properties.edge_geometry_coordinates_atlas
        self.saveFile(array, file)

    def writeGraphPropertyEdgeGeometryAnnotation(self, file = folder +'edge_geometry_annotation.csv'):
        array = self.graph.graph_properties.edge_geometry_annotation
        self.saveFile(array, file)

    def computeLength(self, A, B):
        return numpy.linalg.norm(A-B)

    def computeTotalResistance(self, L, r):
        # with conversion factor 1.63 [um/px]
        return (128. * 1.63 * L) / (numpy.pi * (2 * 1.63 * r)**4)

    def computeEffectiveRadii(self, L_tot, R_tot):
        # with conversion factor 1.63 [um/px]
        return (1/2.) * ((128 * L_tot) / (numpy.pi * R_tot))**(1/4.) * (1 / 1.63)

    def computeEffectiveRadiiAll(self):
        self.edge_length = numpy.zeros(self.graph.num_edges())
        edge_resistance = numpy.zeros(self.graph.num_edges())


        edge_coordinates = self.graph.graph_properties.edge_geometry_coordinates
        seg_length = numpy.linalg.norm(edge_coordinates[1:] - edge_coordinates[:-1], axis=1)

        edge_radii = self.graph.graph_properties.edge_geometry_radii
        seg_radii = (edge_radii[1:] + edge_radii[:-1]) * 0.5

        seg_resistance = self.computeTotalResistance(seg_length, seg_radii)

        v1 = numpy.transpose(self.graph.edge_properties.edge_geometry_indices.get_2d_array((0,1)))

        for k, i, j in zip(range(self.graph.num_edges()), v1[:,0], v1[:,1]-1):
            self.edge_length[k] = numpy.sum(seg_length[i:j])
            edge_resistance[k] = numpy.sum(seg_resistance[i:j])

        self.edge_radii = self.computeEffectiveRadii(self.edge_length, edge_resistance)

    def writeEffectiveRadiiAll(self, file = 'edge_%s.csv'):
        self.computeEffectiveRadiiAll()
        self.saveFile(self.edge_radii, file%'eff_radii')
        self.saveFile(self.edge_length, file%'length')

    def writeAll(self):
        self.writeEdges()
        self.writeEdgePropertyRadii()
        self.writeEdgePropertyRadii_Atlas()
        self.writeEdgePropertyVein()
        self.writeEdgePropertyArtery()
        self.writeEdgePropertyLength()
        self.writeVertices()
        self.writeVertexPropertyCoordinates()
        self.writeVertexPropertyRadii()
        self.writeVertexPropertyCoordinatesAtlas()
        self.writeVertexPropertyRadiiAtlas()
        self.writeVertexPropertyAnnotation()
        self.writeVertexPropertyDistanceToSurface()
        # Add the Tortous Graph properties
        self.writeGraphPropertyEdgeGeometryCoordinates()
        self.writeGraphPropertyEdgeGeometryRadii()
        self.writeGraphPropertyEdgeGeometryIndices()
        self.writeGraphPropertyEdgeGeometryCoordinatesAtlas()
        self.writeGraphPropertyEdgeGeometryArteryRaw()
        self.writeGraphPropertyEdgeGeometryArteryBinary()
        self.writeGraphPropertyEdgeGeometryAnnotation()

class QualityCheck(object):
    def __init__(self, graph):
        self.graph = graph
        self.getRandomSamples()
        self.checkLength()
        self.checkRadii()

    def getRandomSamples(self):
        self.samples = [random.randint(0, self.graph.num_edges()) for i in range(1, 10)]
        print("Random edges to test the data: %s" %(', '.join([str(i) for i in self.samples])))

    def checkLength(self):
        for edge in self.samples:
            l = self.graph.edge_properties.length.get_array()[edge]
            A = numpy.transpose(self.graph.vertex_properties.coordinates.get_2d_array((0,1,2)))[self.graph.get_edges()[edge][0]]
            B = numpy.transpose(self.graph.vertex_properties.coordinates.get_2d_array((0,1,2)))[self.graph.get_edges()[edge][1]]
            l_comp = numpy.linalg.norm(A-B)
            print('%s, %.3f, %.3f'%(edge, l, l_comp))


    def checkRadii(self):
        for edge in self.samples:
            re = self.graph.edge_properties.radii.get_array()[edge]
            rv1 = self.graph.vertex_properties.radii.get_array()[self.graph.get_edges()[edge][0]]
            rv2 = self.graph.vertex_properties.radii.get_array()[self.graph.get_edges()[edge][1]]
            print('%s, %.3f, %.3f, %.3f'%(edge, re, rv1, rv2))

    def checkTortuosCoordinates(self):

        edge_geometry_indices = numpy.transpose(self.graph.edge_properties.edge_geometry_indices.get_2d_array((0,1)))
        a = edge_geometry_indices[:,0][1:]
        b = edge_geometry_indices[:,1][:-1]
        if not numpy.all(a==b):
            print('Non matching edge_geometry_indices')

        vertex_properties_coordinates = numpy.transpose(self.graph.vertex_properties.coordinates.get_2d_array((0,1,2)))
        edge_geometry_coordinates = self.graph.graph_properties.edge_geometry_coordinates
        edges = self.graph.get_edges()

        for k, i, j in zip(range(self.graph.num_edges()), edge_geometry_indices[:,0], edge_geometry_indices[:,1]):
            a = vertex_properties_coordinates[edges[k]]
            b = numpy.array([edge_geometry_coordinates[i], edge_geometry_coordinates[j-1]])
            if not numpy.all(a==b):
                if not numpy.all(a==b[::-1]):
                    print('Non matching tortuos coordinates in edge %s: %s, %s'%(edge,a,b))

if __name__ == '__main__':
    graph = ReadWriteGraph("/home/admin/Ana/MicroBrain/18_vessels_graph.gt")
    #graph.writeAll()
    graph.writeEdgePropertyRadii_Atlas() # i only need to export this right now, as i have all the other csvs 
    QualityCheck(graph.graph)

from dwave.system.samplers import DWaveSampler  # Library to interact with the QPU
from dwave.system.composites import FixedEmbeddingComposite  # Library to embed our problem onto the QPU physical graph
from dwave.system.composites import EmbeddingComposite
from dimod.binary_quadratic_model import BinaryQuadraticModel
import minorminer
import numpy as np


class Sampler(object):
    """
    This module defines a sampler.
    :param num_samps: number of samples
    :type num_samps: int
    """

    def __init__(self, num_copies=1):
        self.endpoint = 'https://cloud.dwavesys.com/sapi'
        self.token = 'DEV-db4d47e5313cf3c52cac31dace7c5080a5ffc46d'
        self.solver = 'DW_2000Q_2_1'
        self.gamma = 1400
        self.chainstrength = 4700
        self.num_copies = num_copies
        self.child = DWaveSampler(endpoint=self.endpoint, token=self.token, solver=self.solver)

    def sample_qubo(self, Q, num_samps=100):
        """
        Sample from the QUBO problem
        :param qubo: QUBO problem
        :type qubo: numpy dictionary
        :return: samples, energy, num_occurrences
        """
        #print(Q)
        self.num_samps = num_samps

        if not hasattr(self, 'sampler'):

            bqm = BinaryQuadraticModel.from_qubo(Q)

            # apply the embedding to the given problem to map it to the child sampler
            __, target_edgelist, target_adjacency = self.child.structure

            # add self-loops to edgelist to handle singleton variables
            source_edgelist = list(bqm.quadratic) + [(v, v) for v in bqm.linear]

            # get the embedding
            embedding = minorminer.find_embedding(source_edgelist, target_edgelist)

            if bqm and not embedding:
                raise ValueError("no embedding found")

            self.sampler = FixedEmbeddingComposite(self.child, embedding)
        response = EmbeddingComposite(DWaveSampler(token="DEV-db4d47e5313cf3c52cac31dace7c5080a5ffc46d")).sample_qubo(Q, num_reads=1000)
        #response = self.sampler.sample_qubo(Q, chain_strength=self.chainstrength, num_reads=self.num_samps)
        #for sample, energy, num_occurrences, chain_break_fraction in list(response.data()):
        #    print(sample, "Energy: ", energy, "Occurrences: ", num_occurrences)
        #print(response.samples()[0,[range(0,71)]])
        #print(response._asdict()['vectors']['energy'])
        #print(response._asdict()['vectors']['num_occurrences'])
        saempeul=np.empty((0,72))
        for sample, in response.data(fields=['sample']):
            saempeul=np.append(saempeul, sample)
        #print(saempeul)
        return saempeul, response._asdict()['vectors']['energy']['data'], \
               response._asdict()['vectors']['num_occurrences']['data']

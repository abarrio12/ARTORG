# ARTORG

This repository contains two vascular graph projects: **ParisGraph** and **XiangJi**.

---

## ParisGraph

The `ParisGraph` project starts from the reconstruction of the `18_halfbrain` vascular graph.

From this graph, six 3D boxes are extracted:

- 3 boxes from the hippocampal area
- 3 boxes from the somatomotor cortex region

The workflow includes:

1. cutting the six regional graphs from the original half-brain graph;
2. analysing each of the six graphs separately;
3. comparing the hippocampal and somatomotor regions to study vascular differences between brain areas.

---

## XiangJi

The `XiangJi` project reconstructs two graphs from `.mat` files into `.pkl` format:

- one crop graph (ML20180815_240_c5o1_578.mat)
- one whole-brain graph (WholeBrain_ML_2018_08_15_whole_brain_graph)

These graphs are converted into an MVN-style format, making them compatible with the ParisGraph code structure and analysis tools.

However, there are still some dataset-specific differences. For example, the XiangJi whole-brain graph is currently not annotated by brain region (no nkind).

---

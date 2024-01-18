"""
The IRIS model is the function model for this process of base calling.

This model include 6 part of sub-models, of which, 4 public sub-models are used to transform images into barcode,
including 1) importing images and storing them in a 3D tensor data structure; 2) detecting blobs, then transforming
theme into the bases and calculating their error rate by a test of binomial distribution under 50% success ratio; 3)
connecting the called bases in a same location in different cycles, storing them into the barcode sequences; 4)
reformatting the result and outputting. And, 2 private sub-models are used to register images among different cycles,
and transform blobs into bases.
"""
# Third-party notices

The ACoPSe research implementation includes code adapted from prior work and uses separately installed Python dependencies. This document records the observed source lineage and the notices that apply to the identified upstream portions.

## Vendored clustering code

The retained files under `torch_clustering/` come from Zhizhong Huang's `torch_clustering` project. The upstream project publishes an MIT notice. Its copyright line contains placeholders, which are preserved in the text below. The exact vendored revision has not been established. [Upstream source](https://github.com/Hzzone/torch_clustering/tree/master/torch_clustering) · [Upstream MIT notice](https://raw.githubusercontent.com/Hzzone/torch_clustering/master/LICENSE)

`torch_clustering/kmeans/kmeans_plus_plus.py` retains identifiable scikit-learn material adapted to PyTorch, including K-means++ initialization and `stable_cumsum`. The applicable scikit-learn BSD 3-Clause notice is included below. The linked source versions document the overlap. The exact originating revision has not been established. [K-means initializer](https://github.com/scikit-learn/scikit-learn/blob/1.1.2/sklearn/cluster/_kmeans.py#L142-L222) · [Cumulative-sum helper](https://github.com/scikit-learn/scikit-learn/blob/1.0.2/sklearn/utils/extmath.py) · [BSD 3-Clause notice](https://raw.githubusercontent.com/scikit-learn/scikit-learn/1.0.2/COPYING)

## Retained code with unverified redistribution terms

| Source lineage | Retained material | Status |
| --- | --- | --- |
| CIRH, by Lei Zhu, Xize Wu, Jingjing Li, Zheng Zhang, Weili Guan, and Heng Tao Shen | Adapted MAT-loading code, hash-network and generator components, training and evaluation orchestration, affinity and quantization code, and retrieval utilities. | No explicit license grant was found in the inspected public repository. Redistribution terms for the retained portions are unverified. [CIRH repository](https://github.com/XizeWu/CIRH) · [CIRH paper](https://doi.org/10.1109/TKDE.2022.3218656) |
| UCCH | `NCE/NCEAverage.py` and `NCE/NCECriterion.py` match the published UCCH versions after whitespace normalization. Earlier CMC code contains matching portions of the implementation. The CMC BSD 2-Clause notice below covers the identified CMC lineage. | No explicit license grant was found in the inspected UCCH repository. Redistribution terms for the UCCH versions are unverified. [UCCH source](https://github.com/penghu-cs/UCCH/tree/main/NCE) · [CMC notice](https://raw.githubusercontent.com/HobbitLong/CMC/master/LICENSE) |
| `lemniscate.pytorch` alias sampler | `NCE/alias_multinomial.py` closely matches the upstream `AliasMethod` implementation and cites the same Harvard algorithm reference. | No explicit license grant was found in the inspected public repository. Redistribution terms for the implementation are unverified. [Upstream source](https://github.com/zhirongw/lemniscate.pytorch/blob/master/lib/alias_multinomial.py) · [Algorithm reference](https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/) |

The `NCE/` files are retained as legacy research material. The active trainer uses `lightly.loss.DCLLoss` from the separately installed Lightly package. The validated environment used Lightly 1.4.3, whose package metadata lists the MIT license. See the environment guide for the other runtime dependencies. [Lightly package](https://pypi.org/project/lightly/1.4.3/) · [Environment guide](docs/environment.md)

## License texts

### Hzzone torch_clustering

```text
MIT License

Copyright (c) [year] [fullname]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### scikit-learn

```text
BSD 3-Clause License

Copyright (c) 2007-2021 The scikit-learn developers.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

### Contrastive Multiview Coding

```text
BSD 2-Clause License

Copyright (c) 2020, Yonglong Tian
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

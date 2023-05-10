#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import io
from setuptools import setup, find_packages

__version__ = '0.1.1.dev1'

with io.open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

setup(
    name='topoworld',
    version=__version__,
    author='parl_dev',
    author_email='',
    description='TopoWorld: Topological game environments for benchmarking Reinforcement Learning',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/milosen/topological-labyrinths-rl',
    license="GPLv3",
    packages=["topoworld"],
    package_data={'topoworld': []},
    install_requires=[
        'tqdm',
        'rich',
        'pyyaml',
        'miniworld==2.0.0',
        'matplotlib',
        'gymnasium',
        'stable_baselines3>=2.0.0a1',
        'sb3_contrib>=2.0.0a1',
        'tensorboard',
        'seaborn',
        'jupyter'
    ],
    extras_require={},
    zip_safe=False,
)

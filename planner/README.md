# DAPPLE Planner

## Getting Started
### Install

#### From Rust Cargo, as a Rust crate
```bash
cargo install HPGO
```
`TODO: publish to Cargo crates after open source`

#### From Python PyPI, as a Python3 package
```bash
pip3 install HPGO
```
`TODO: publish to Python package index after open source`

### Build from source
```bash
rustup default nightly
cargo build --release
maturin build --release
pip3 install xxx.whl
```

### Use
```python
# Import HPGO Python API
import HPGO
# Construct the Conductor object
c = conductor_from_torch_graph_and_seps("./profiles/xlnet/graph.txt", 64, 512, [8, 16])
c.py_orchestrate()
```

## License

This project (DAPPLE Planner) is open sourced under the terms of BSD-3-Clause, details of which can be found in the [`LICENSE.md`](LICENSE.md) file

The file `src/input/torch_graph_py.rs` contains Python source code from [PipeDream](https://github.com/msr-fiddle/pipedream), which is licensed under the MIT License.

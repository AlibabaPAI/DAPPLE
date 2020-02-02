extern crate HPGO;
use HPGO::environment::network;
use HPGO::input::*;
use HPGO::orchestration::*;

#[test]
fn test_orchestrate_hierarchical_compute_plan() {
    let mut c = orchestrate_hierarchical::HierarchicalConductor::new_from_torch_graph(
        "./profiles/xlnet/graph.txt",
        2,
        128,
    );
    c.compute_plan_hierarchical(8, 1, network::GIGABYTE, true);
    for i in 0..c.ctx.len() {
        for j in 0..c.ctx[i].len() {
            for k in 0..c.ctx[i][j].len() {
                println!("{} {} {} | {:?}", i, j, k, c.ctx[i][j][k]);
            }
        }
    }
}

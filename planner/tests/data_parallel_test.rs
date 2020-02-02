extern crate HPGO;
use HPGO::environment::*;
use HPGO::parallelism::data_parallel;

#[test]
fn test_all_reduce_time() {
    let d16 = device::Devices::new(16, vec![8, 16]);
    let in_machine: &[u32] = &[0, 1, 2, 3, 4, 5, 6, 7];
    let mut calculated_time = data_parallel::all_reduce_time(
        &d16,
        &in_machine.iter().cloned().collect(),
        network::GIGABYTE,
    );
    let mut expected_time: f64 =
        network::GIGABYTE * 2.0 * 7.0 / 8.0 / nvlink::BANDWIDTH_NVLINK_P2P / 4.0;
    println!(
        "calculated at: {}, expected: {}",
        calculated_time, expected_time
    );
    assert_eq!((calculated_time - expected_time).abs() < 0.001, true);

    let between_machine: &[u32] = &[1, 2, 3, 4, 5, 6, 7, 8];
    calculated_time = data_parallel::all_reduce_time(
        &d16,
        &between_machine.iter().cloned().collect(),
        network::GIGABYTE,
    );
    expected_time = network::GIGABYTE * 2.0 * 7.0 / 8.0 / ethernet::BANDWIDTH_ETHERNET_NCCL;
    println!(
        "calculated at: {}, expected: {}",
        calculated_time, expected_time
    );
    assert_eq!((calculated_time - expected_time).abs() < 0.001, true);
}

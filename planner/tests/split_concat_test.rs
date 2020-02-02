extern crate HPGO;
use HPGO::environment::*;
use HPGO::parallelism::split_concat;

fn t_split_concat_all2all_time(
    d: &device::Devices,
    from: &[u32],
    to: &[u32],
    size: f64,
    expect: f64,
) {
    let calculated_time = split_concat::split_concat_all2all_time(
        d,
        &from.iter().cloned().collect(),
        &to.iter().cloned().collect(),
        size,
    );
    println!("calculated at: {}, expected: {}", calculated_time, expect);
    assert_eq!((calculated_time - expect).abs() < 0.001, true);
}

#[test]
fn test_split_concat_time() {
    let d16 = device::Devices::new(16, vec![8, 16]);
    t_split_concat_all2all_time(
        &d16,
        &[0, 1, 2, 3],
        &[4, 5, 6, 7],
        network::GIGABYTE,
        network::GIGABYTE / nvlink::BANDWIDTH_NVLINK_P2P,
    );
    t_split_concat_all2all_time(
        &d16,
        &[0, 1, 8, 9],
        &[10, 11, 12, 13],
        network::GIGABYTE,
        0.5 * network::GIGABYTE / ethernet::BANDWIDTH_ETHERNET_P2P,
    );
}

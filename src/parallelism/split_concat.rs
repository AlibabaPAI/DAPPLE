use environment::*;
use std::collections::BTreeSet;

pub fn split_concat_all2all_time(
    d: &device::Devices,
    from: &BTreeSet<u32>,
    to: &BTreeSet<u32>,
    size: f64,
) -> f64 {
    let split_concat_strict_cross_machine_check = true;
    if split_concat_strict_cross_machine_check {
        let b_to_cross = d.is_cross_machine_within(to);
        if !b_to_cross {
            let mut cross_machine_gids: BTreeSet<u32> = BTreeSet::new();
            for s in from {
                if d.is_cross_machine_from_to(&[*s].iter().cloned().collect(), to) {
                    cross_machine_gids.insert(*s);
                }
            }
            // println!("cross_machine_gids: {:?}", cross_machine_gids);
            if cross_machine_gids.len() > 0 {
                size / from.len() as f64 * cross_machine_gids.len() as f64
                    / ethernet::BANDWIDTH_ETHERNET_P2P
            } else {
                size / nvlink::BANDWIDTH_NVLINK_P2P
            }
        } else {
            match d.is_cross_machine_from_to(from, to) {
                true => size / ethernet::BANDWIDTH_ETHERNET_NCCL,
                false => size / nvlink::BANDWIDTH_NVLINK_P2P,
            }
        }
    } else {
        match d.is_cross_machine_from_to(from, to) {
            true => size / ethernet::BANDWIDTH_ETHERNET_NCCL,
            false => size / nvlink::BANDWIDTH_NVLINK_P2P,
        }
    }
}

pub fn split_concat_time(
    d: &device::Devices,
    from: &BTreeSet<u32>,
    to: &BTreeSet<u32>,
    size: f64,
) -> f64 {
    unimplemented!()
}

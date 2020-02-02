extern crate HPGO;
use HPGO::environment::*;

#[test]
fn test_new_device() {
    // should not crash
    let _d = device::Devices::new(16, vec![8, 16]);
}

/// test helper to perform cross_machine test with (from, to)
fn t_from_to(d: &device::Devices, from: &[u32], to: &[u32], val: bool) {
    assert_eq!(
        d.is_cross_machine_from_to(
            &from.iter().cloned().collect(),
            &to.iter().cloned().collect()
        ),
        val
    );
}

/// test helper to perform cross_machine test within (gids)
fn t_within(d: &device::Devices, gids: &[u32], val: bool) {
    assert_eq!(
        d.is_cross_machine_within(&gids.iter().cloned().collect()),
        val
    );
}

#[test]
fn test_cross_machine_from_to() {
    let d16 = device::Devices::new(16, vec![8, 16]);
    t_from_to(&d16, &[0, 1, 2, 3], &[4, 5, 6, 7], false);
    t_from_to(&d16, &[0, 1, 2, 3], &[5, 6, 7, 8], true);
    t_from_to(&d16, &[0, 1, 2, 3], &[8], true);
    t_from_to(&d16, &[0, 1, 2, 3], &[5], false);

    let d2 = device::Devices::new(2, vec![1, 2]);
    t_from_to(&d2, &[0], &[1], true);
}

#[test]
fn test_cross_machine_within() {
    let d16 = device::Devices::new(16, vec![8, 16]);
    t_within(&d16, &[1, 2, 3, 4], false);
    t_within(&d16, &[0, 1, 2, 3, 4, 5, 6, 7], false);
    t_within(&d16, &[7, 8], true);
}

/// test helper to perform next_cards test
fn t_next_cards(d: &device::Devices, bs: &[usize], need: u32, replica: u32, _expect: Vec<u32>) {
    let mut bitset: Vec<bool> = vec![];
    for b in bs {
        match b {
            0 => {
                bitset.push(false);
            }
            1 => {
                bitset.push(true);
            }
            _ => {
                panic!("undefined sequence");
            }
        }
    }
    match replica {
        1 => {
            println!("{:?}", d.next_cards(bitset, need));
        }
        _ => {
            println!("{:?}", d.next_cards_with_replica(bitset, need, replica));
        }
    }
}

#[test]
fn test_next_cards() {
    let d8 = device::Devices::new(8, vec![4, 8]);
    println!("request 4 cards from a new 2m8g environment");
    t_next_cards(&d8, &[0, 0, 0, 0, 0, 0, 0, 0], 4, 1, vec![]);
    println!("request 4 cards from a 2m8g environment with first card occupied");
    t_next_cards(&d8, &[1, 0, 0, 0, 0, 0, 0, 0], 4, 1, vec![]);
    println!("request 3 cards from a 2m8g environment with first card occupied");
    t_next_cards(&d8, &[1, 0, 0, 0, 0, 0, 0, 0], 3, 1, vec![]);
}

#[test]
fn test_next_cards_with_replica() {
    let d8 = device::Devices::new(8, vec![4, 8]);
    println!("request 4x2 cards from a new 2m8g environment");
    t_next_cards(&d8, &[0, 0, 0, 0, 0, 0, 0, 0], 4, 2, vec![]);
    let d16 = device::Devices::new(16, vec![8, 16]);
    println!("request 8 cards from a new 2m16g environment, twice");
    t_next_cards(
        &d16,
        &[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        8,
        1,
        vec![],
    );
    t_next_cards(
        &d16,
        &[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        8,
        1,
        vec![],
    );
    println!("request 4x3 cards from a new 2m16g environment, twice");
    t_next_cards(
        &d16,
        &[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        4,
        3,
        vec![],
    );
    t_next_cards(
        &d16,
        &[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        4,
        3,
        vec![],
    );
}

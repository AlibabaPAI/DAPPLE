use environment::network::GIGABYTE;
use pyo3::prelude::*;
use std::cmp::{max, min};
use std::collections::{BTreeSet, HashMap};
use std::iter::FromIterator;

#[derive(Debug, Clone)]
pub struct ReturnDevices {
    pub strategy: AllocationStrategy,
    pub occupied: Vec<bool>,
    pub gids: BTreeSet<u32>, // GPU id array, BTreeSet because its hashable
}

/// Allocation Strategy used to allocate requested new cards
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum AllocationStrategy {
    FreshFirst,
    AppendFirst,
    ScatterFirst,
    GapsFirst,
    Fusion,
}

#[pyclass]
#[derive(Debug, Clone)]
pub struct Devices {
    pub num_machines: u32,
    pub num_gpus: u32,
    pub occupied: Vec<bool>,
    pub seps: Vec<u32>,
}

// define the GPU memory to be constant for now
pub const GPU_MEMORY: f64 = 16.0 * GIGABYTE;

impl Devices {
    pub fn new(n: u32, seps: Vec<u32>) -> Devices {
        let mut dbs: Vec<bool> = vec![];
        for _i in 0..n {
            dbs.push(false);
        }
        Devices {
            num_machines: seps.len() as u32,
            num_gpus: n,
            seps: seps,
            occupied: dbs,
        }
    }

    pub fn all_gpus(&self) -> BTreeSet<u32> {
        let vec_all_gids: Vec<u32> = (0..self.num_gpus).collect();
        let all_gids: BTreeSet<u32> = vec_all_gids.iter().cloned().collect();
        all_gids
    }

    pub fn is_cross_machine_from_to(&self, from: &BTreeSet<u32>, to: &BTreeSet<u32>) -> bool {
        let min_from = from.iter().min().expect("min_from error");
        let max_from = from.iter().max().expect("max_from error");
        let min_to = to.iter().min().expect("min_to error");
        let max_to = to.iter().max().expect("max_to error");
        let global_min = min(min_from, min_to);
        let global_max = max(max_from, max_to);

        // println!("Global Min {} Max {}", global_min, global_max);

        let mut l: u32 = 0;
        for r in &self.seps {
            if l <= *global_min && *global_min < *r && l <= *global_max && *global_max < *r {
                return false;
            }
            l = *r;
        }

        true
    }

    pub fn is_cross_machine_within(&self, gids: &BTreeSet<u32>) -> bool {
        if gids.is_empty() {
            panic!("[device] is_cross_machine_within: gids is NULL");
        }
        let min_id = gids.iter().min().expect("min_id error");
        let max_id = gids.iter().max().expect("max_id error");

        let mut l: u32 = 0;
        for r in &self.seps {
            if l <= *min_id && *min_id < *r && l <= *max_id && *max_id < *r {
                return false;
            }
            l = *r;
        }

        true
    }

    fn next_cards_with_strategy(
        &self,
        bs: Vec<bool>,
        need: u32,
        mut machine_availability: Vec<(bool, u32)>, // already cloned
        strategy: AllocationStrategy,
    ) -> Option<ReturnDevices> {
        // println!("[device]\t next_cards_with_strategy: {:?}", strategy);
        // FF
        let mut t_ret: ReturnDevices = ReturnDevices {
            strategy: strategy,
            occupied: bs,
            gids: BTreeSet::new(),
        };
        let mut n_ret = need;
        match &t_ret.strategy {
            AllocationStrategy::ScatterFirst => {
                while n_ret > 0 {
                    for i in 0..machine_availability.len() {
                        if machine_availability[i].1 > 0 && n_ret > 0 {
                            let mut j: u32 = match i {
                                0 => 0,
                                _ => *&self.seps[i - 1],
                            };
                            while t_ret.occupied[j as usize] == true {
                                j += 1;
                            }
                            t_ret.occupied[j as usize] = true;
                            t_ret.gids.insert(j);
                            n_ret -= 1;
                            machine_availability[i].1 -= 1;
                        }
                    }
                }
            }
            AllocationStrategy::GapsFirst => {
                unimplemented!();
            }
            _ => {
                let need_fresh: bool = match &t_ret.strategy {
                    AllocationStrategy::FreshFirst => true,
                    AllocationStrategy::AppendFirst => false,
                    _ => panic!(),
                };
                for i in 0..machine_availability.len() {
                    if machine_availability[i].0 == need_fresh {
                        let mut j: u32 = match i {
                            0 => 0,
                            _ => self.seps[i - 1],
                        };
                        if machine_availability[i].1 >= n_ret {
                            // TODO: This line has a bug, which causes index overflow for j
                            // println!("[device]\t testing j = {}", j);
                            while t_ret.occupied[j as usize] == true
                                && j < (t_ret.occupied.len() - 1) as u32
                            {
                                j += 1;
                            }
                            while n_ret > 0 {
                                t_ret.occupied[j as usize] = true;
                                t_ret.gids.insert(j);
                                j += 1;
                                n_ret -= 1;
                            }
                        } else {
                            while t_ret.occupied[j as usize] == true {
                                j += 1;
                            }
                            while j < *&self.seps[i] {
                                t_ret.occupied[j as usize] = true;
                                t_ret.gids.insert(j);
                                j += 1;
                                n_ret -= 1;
                            }
                        }
                    }
                }
            }
        }

        if n_ret == 0 {
            Some(t_ret)
        } else {
            None
        }
    }

    pub fn next_cards(&self, bs: Vec<bool>, need: u32) -> Vec<ReturnDevices> {
        let mut machine_availability: Vec<(bool, u32)> = vec![];
        let mut start = 0;
        let mut left_total = 0;
        let mut t_ret: Vec<ReturnDevices> = vec![];
        for s in &self.seps {
            let mut fresh = true;
            let mut left = 0;
            for i in start..*s {
                match bs[i as usize] {
                    true => {
                        fresh = false;
                    }
                    false => {
                        left += 1;
                    }
                }
            }
            machine_availability.push((fresh, left));
            left_total += left;
            start = *s;
        }

        assert_eq!((left_total >= need), true);

        let mut exist: HashMap<BTreeSet<u32>, ReturnDevices> = HashMap::new();

        for s in vec![
            AllocationStrategy::FreshFirst,
            AllocationStrategy::AppendFirst,
            AllocationStrategy::ScatterFirst,
        ] {
            let t =
                (&self).next_cards_with_strategy(bs.clone(), need, machine_availability.clone(), s);
            match t {
                Some(result) => {
                    exist.insert(result.gids.clone(), result);
                }
                None => {}
            }
        }

        for (_k, v) in exist {
            t_ret.push(v);
        }

        return t_ret;
    }

    pub fn next_cards_with_replica_helper(
        &self,
        bs: Vec<bool>,
        need: u32,
        replica: u32,
    ) -> Vec<Vec<ReturnDevices>> {
        let single_round_result = self.next_cards(bs, need);
        let mut res: Vec<Vec<ReturnDevices>> = vec![];
        if replica == 1 {
            for sr in single_round_result {
                res.push(vec![sr]);
            }
            return res;
        }
        for sr in single_round_result {
            let cur_bs = sr.occupied.clone();
            let sub_sr = self.next_cards_with_replica_helper(cur_bs, need, replica - 1);
            for ssr in sub_sr {
                let mut new_ssr = ssr;
                new_ssr.insert(0, sr.clone());
                res.push(new_ssr);
            }
        }
        return res;
    }

    pub fn next_cards_with_replica(
        &self,
        bs: Vec<bool>,
        need: u32,
        replica: u32,
    ) -> Vec<ReturnDevices> {
        let mut res: Vec<ReturnDevices> = vec![];
        let mut exist: HashMap<BTreeSet<u32>, ReturnDevices> = HashMap::new();
        let separate_result = self.next_cards_with_replica_helper(bs, need, replica);
        for s in separate_result {
            let mut dim_bs: Vec<bool> = vec![];
            let mut dim_gids: Vec<u32> = vec![];
            for ydim_s in s {
                dim_bs = ydim_s.occupied;
                dim_gids.extend(&ydim_s.gids);
            }
            // NOTE: the following two lines are extremely ugly and a uttermost disrespect to the Rust Core team
            // But fsck mate I don't want to deal with borrow checker and ownership anymore
            let gids: BTreeSet<u32> = BTreeSet::from_iter(dim_gids.iter().cloned());
            let gids_signature: BTreeSet<u32> = BTreeSet::from_iter(dim_gids.iter().cloned());
            let sub_bs = dim_bs;
            exist.insert(
                gids_signature,
                ReturnDevices {
                    strategy: AllocationStrategy::Fusion,
                    occupied: sub_bs,
                    gids: gids,
                },
            );
        }

        for (_k, v) in exist {
            res.push(v);
        }

        return res;
    }
}

use std::ops::RangeInclusive;

#[derive(Debug, Clone, Eq)]
pub struct Set {
    pub(crate) ranges: Vec<RangeInclusive<i64>>,
}

impl Set {
    pub fn new(base: RangeInclusive<i64>) -> Set {
        Set { ranges: vec![base] }
    }

    #[cfg(test)]
    pub fn multi(bases: &[RangeInclusive<i64>]) -> Set {
        Set {
            ranges: bases.to_vec(),
        }
    }

    pub fn empty() -> Set {
        Set { ranges: Vec::new() }
    }

    pub fn merge(&mut self, other: &Set) {
        for range in &other.ranges {
            self.insert_and_merge(range.clone())
        }
    }

    pub fn and(&mut self, other: &Set) {
        let mut new_ranges = Vec::new();
        for range in &self.ranges {
            for other_range in &other.ranges {
                if range.start() <= other_range.end() && range.end() >= other_range.start() {
                    new_ranges.push(
                        *range.start().max(other_range.start())
                            ..=*range.end().min(other_range.end()),
                    );
                }
            }
        }
        self.ranges = new_ranges;
    }

    pub fn insert_and_merge(&mut self, range: RangeInclusive<i64>) {
        let mut touched = Vec::new();
        for (i, other) in self.ranges.iter().enumerate() {
            if range.start().saturating_sub(1) <= *other.end()
                && range.end().saturating_add(1) >= *other.start()
            {
                touched.push((i, other.clone()))
            }
        }
        if touched.is_empty() {
            self.ranges.push(range)
        } else {
            touched.push((usize::MAX, range));
            let min = *touched.iter().map(|(_, r)| r.start()).min().unwrap();
            let max = *touched.iter().map(|(_, r)| r.end()).max().unwrap();
            for (i, _) in touched.into_iter().rev() {
                if i < usize::MAX {
                    self.ranges.remove(i);
                }
            }
            self.ranges.push(min..=max);
        }
    }

    fn remove_range_from_range(
        range: RangeInclusive<i64>,
        removal_range: &RangeInclusive<i64>,
    ) -> Vec<RangeInclusive<i64>> {
        let (range_start, range_end) = (*range.start(), *range.end());
        let (removal_start, removal_end) = (*removal_range.start(), *removal_range.end());

        if removal_start > range_end || removal_end < range_start {
            // Case 1: No overlap
            vec![range]
        } else if removal_start <= range_start && removal_end >= range_end {
            // Case 2: Complete overlap
            vec![]
        } else if removal_start > range_start && removal_end < range_end {
            // Case 3: Partial overlap within the range, split into two
            vec![
                RangeInclusive::new(range_start, removal_start - 1),
                RangeInclusive::new(removal_end + 1, range_end),
            ]
        } else if removal_start <= range_start {
            // Overlaps at the start
            vec![RangeInclusive::new(removal_end + 1, range_end)]
        } else {
            // Overlaps at the end
            vec![RangeInclusive::new(range_start, removal_start - 1)]
        }
    }

    pub fn remove_range(&mut self, range: RangeInclusive<i64>) {
        self.ranges = self
            .ranges
            .drain(..)
            .flat_map(|r| Self::remove_range_from_range(r, &range))
            .collect();
    }

    pub fn remove_int(&mut self, to_remove: i64) {
        self.remove_range(to_remove..=to_remove)
    }

    pub fn is_subset(&self, other: &Set) -> bool {
        'outer: for range in &self.ranges {
            for other_range in &other.ranges {
                if range.start() >= other_range.start() && range.end() <= other_range.end() {
                    continue 'outer;
                }
            }
            return false;
        }
        true
    }

    fn format_range(range: &RangeInclusive<i64>) -> String {
        match (range.start(), range.end()) {
            (&i64::MIN, &i64::MAX) => "..".to_string(),
            (&i64::MIN, max) => {
                format!("..{max}")
            }
            (min, &i64::MAX) => {
                format!("{min}..")
            }
            (min, max) => {
                format!("{min}..{max}")
            }
        }
    }

    pub fn format(&self) -> String {
        if self.ranges.len() == 1 {
            Self::format_range(&self.ranges[0])
        } else {
            let content = self
                .ranges
                .iter()
                .map(Self::format_range)
                .collect::<Vec<_>>()
                .join(", ");
            format!("[{content}]")
        }
    }

    pub fn min(&self) -> i64 {
        self.ranges
            .iter()
            .map(|r| *r.start())
            .min()
            .unwrap_or(i64::MIN)
    }

    pub fn max(&self) -> i64 {
        self.ranges
            .iter()
            .map(|r| *r.end())
            .max()
            .unwrap_or(i64::MAX)
    }
}

impl PartialEq for Set {
    fn eq(&self, other: &Self) -> bool {
        if self.ranges == other.ranges {
            return true;
        }
        if self.ranges.len() != other.ranges.len() {
            return false;
        }
        for range in &self.ranges {
            if !other.ranges.contains(range) {
                return false;
            }
        }
        for range in &other.ranges {
            if !self.ranges.contains(range) {
                return false;
            }
        }
        true
    }
}

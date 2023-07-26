def merge_ranges(set1, set2):
    # Concatenate both sets of ranges
    all_ranges = set1 + set2

    # Sort the ranges based on their start values
    sorted_ranges = sorted(all_ranges, key=lambda x: x[0])

    merged_ranges = []
    prev_start, prev_end = sorted_ranges[0]

    for start, end in sorted_ranges[1:]:
        if start <= prev_end:
            # Merge overlapping or adjacent ranges
            prev_end = max(prev_end, end)
        else:
            # Add the non-overlapping range to the merged list
            merged_ranges.append((prev_start, prev_end))
            prev_start, prev_end = start, end

    # Add the last range to the merged list
    merged_ranges.append((prev_start, prev_end))

    # Remove duplicates from merged_ranges
    merged_ranges = list(set(merged_ranges))

    # Sort the merged ranges based on their start values
    merged_ranges.sort(key=lambda x: x[0])

    return merged_ranges

# Example usage:
set1 = [(1, 5), (10, 15), (20, 25)]
set2 = [(4, 8), (12, 18), (22, 30)]
result = merge_ranges(set1, set2)
print(result)  # Output: [(1, 8), (10, 18), (20, 30)]

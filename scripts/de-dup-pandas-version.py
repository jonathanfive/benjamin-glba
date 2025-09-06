import os
import hashlib
import pandas as pd
from pathlib import Path
from difflib import SequenceMatcher
from collections import defaultdict


def calculate_file_hash(filepath):
    """Calculate MD5 hash of file content"""
    hash_md5 = hashlib.md5()
    try:
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except Exception as e:
        print(f"Error hashing {filepath}: {e}")
        return None


def similarity_ratio(a, b):
    """Calculate similarity ratio between two strings"""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def find_duplicates():
    base_dir = Path(".")

    # Scan all PDF files (both .pdf and .PDF extensions)
    all_files = []
    for pattern in ["*.pdf", "*.PDF"]:
        for pdf_file in base_dir.rglob(pattern):
            if pdf_file.is_file():
                all_files.append(pdf_file)

    print(f"Found {len(all_files)} PDF files to analyze")

    # Calculate hashes and collect metadata
    file_data = []
    hash_groups = defaultdict(list)

    for filepath in all_files:
        file_hash = calculate_file_hash(filepath)
        if file_hash:
            file_info = {
                'filepath': str(filepath),
                'filename': filepath.name,
                'directory': str(filepath.parent),
                'size': filepath.stat().st_size,
                'hash': file_hash
            }
            file_data.append(file_info)
            hash_groups[file_hash].append(file_info)

    # Find exact duplicates (same hash)
    exact_duplicates = {k: v for k, v in hash_groups.items() if len(v) > 1}

    # Find similar filenames
    similar_files = []
    processed = set()

    for i, file1 in enumerate(file_data):
        if file1['filename'] in processed:
            continue

        similar_group = [file1]
        for j, file2 in enumerate(file_data[i + 1:], i + 1):
            if file2['filename'] in processed:
                continue

            # Check filename similarity
            ratio = similarity_ratio(file1['filename'], file2['filename'])
            if ratio > 0.8:  # 80% similar
                similar_group.append(file2)
                processed.add(file2['filename'])

        if len(similar_group) > 1:
            similar_files.append(similar_group)
        processed.add(file1['filename'])

    return exact_duplicates, similar_files, file_data


def generate_report(exact_duplicates, similar_files, file_data):
    """Generate comprehensive deduplication report"""

    total_files = len(file_data)
    total_exact_dupes = sum(len(group) - 1 for group in exact_duplicates.values())

    print("\n" + "=" * 60)
    print("GLBA DOCUMENT COLLECTION - DEDUPLICATION REPORT")
    print("=" * 60)
    print(f"Total files analyzed: {total_files}")
    print(f"Exact duplicates found: {total_exact_dupes}")
    print(f"Similar filename groups: {len(similar_files)}")
    print(f"Estimated unique files: {total_files - total_exact_dupes}")

    # Create comprehensive DataFrame
    df_all = pd.DataFrame(file_data)

    # Directory breakdown
    print(f"\nFiles by source directory:")
    dir_summary = df_all.groupby('directory').agg({
        'filename': 'count',
        'size': ['mean', 'sum']
    }).round(2)
    print(dir_summary)

    print("\n" + "-" * 40)
    print("EXACT DUPLICATES (same content)")
    print("-" * 40)

    duplicate_summary = []
    for i, (hash_val, files) in enumerate(exact_duplicates.items(), 1):
        print(f"\nDuplicate group {i} ({len(files)} files):")
        case_name = files[0]['filename'].split(' - ')[-1].replace('.pdf', '').replace('.PDF', '') if ' - ' in files[0][
            'filename'] else files[0]['filename']

        for file_info in files:
            print(f"  {file_info['filepath']}")
            duplicate_summary.append({
                'group': i,
                'case_name': case_name,
                'filepath': file_info['filepath'],
                'directory': file_info['directory'],
                'size': file_info['size']
            })

    print("\n" + "-" * 40)
    print("SIMILAR FILENAMES (potential duplicates)")
    print("-" * 40)

    for i, group in enumerate(similar_files, 1):
        print(f"\nSimilar group {i} ({len(group)} files):")
        for file_info in group:
            print(f"  {file_info['filepath']}")

    # Save comprehensive analysis
    df_all.to_csv('all_files_inventory.csv', index=False)

    if duplicate_summary:
        df_duplicates = pd.DataFrame(duplicate_summary)
        df_duplicates.to_csv('exact_duplicates_analysis.csv', index=False)

    # Create removal suggestions with smart prioritization
    removal_suggestions = []
    for files in exact_duplicates.values():
        # Priority order: federal > WestLaw > NexisUni > HeinOnline
        priority_order = ['court-cases/federal', 'WestLaw', 'NexisUni', 'HeinOnline']

        # Sort files by priority
        def get_priority(file_info):
            for i, priority_dir in enumerate(priority_order):
                if priority_dir in file_info['directory']:
                    return i
            return len(priority_order)

        sorted_files = sorted(files, key=get_priority)
        keep = sorted_files[0]  # Keep highest priority
        remove = sorted_files[1:]  # Remove the rest

        for f in remove:
            removal_suggestions.append({
                'action': 'remove_exact_duplicate',
                'filepath': f['filepath'],
                'keep_instead': keep['filepath'],
                'reason': f'identical_to_{keep["directory"].replace("/", "_")}_version'
            })

    if removal_suggestions:
        df_suggestions = pd.DataFrame(removal_suggestions)
        df_suggestions.to_csv('duplicate_removal_suggestions.csv', index=False)

        print(f"\nRemoval suggestions: {len(removal_suggestions)} files can be safely removed")

    print(f"\nFiles saved:")
    print(f"  - all_files_inventory.csv (complete file listing)")
    if duplicate_summary:
        print(f"  - exact_duplicates_analysis.csv (duplicate groups)")
    if removal_suggestions:
        print(f"  - duplicate_removal_suggestions.csv (removal recommendations)")

    return total_files - total_exact_dupes


if __name__ == "__main__":
    exact_duplicates, similar_files, file_data = find_duplicates()
    unique_count = generate_report(exact_duplicates, similar_files, file_data)
    print(f"\nFinal count: {unique_count} unique documents for GLBA model training")
import argparse
import os
import logging
import time
import sys
import shutil
import subprocess
from types import SimpleNamespace
from collections import namedtuple
import datetime
import filecmp
import copy

def is_media_file(filename, args):
    img_exts = [".jpg", "jpeg", ".png", ".cr2", ".bmp", ".tif"]
    vid_exts = [".mov", ".mts", ".mp4", ".mpg", ".avi", ".flv", ".mkv", ".wmv", ".m4v"]
    exts = []
    if not args.no_images:
        exts.extend(img_exts)
    if not args.no_videos:
        exts.extend(vid_exts)
    return any([filename.lower().endswith(ext) for ext in exts])

def human_redable_size(num, suffix='B'):
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)

def parse_fs_media_file_info_paths(directory, args):
    file_infos = []
    non_media_extensions = set()
    for dirpath, dirnames, filenames in os.walk(args.directory):
        for filename in filenames:
            if is_media_file(filename, args):
                file_infos.append(
                    SimpleNamespace(path=os.path.join(dirpath, filename)))
            else:
                non_media_extensions.add(os.path.splitext(filename)[1])
    return file_infos, non_media_extensions

def progressbar(it, prefix="", size=60, file=sys.stdout):
    count = len(it)
    def show(j):
        x = int(size*j/count)
        file.write("%s[%s%s] %i/%i\r" % (prefix, "#"*x, "."*(size-x), j, count))
        file.flush()
    show(0)
    for i, item in enumerate(it):
        yield item
        show(i+1)
    file.write("\n")
    file.flush()

def prepare_media_file_infos(file_infos,
                             naive_dups=False, list_duplicates=False,
                             mts_encode_sh=None, tmp_dir=None,
                             list_encoded=False):
    def handle_duplicate(finfo, duplicates, dup_size_table, naive_dups):
        size = os.stat(finfo.path).st_size
        if size not in dup_size_table:
            dup_size_table[size] = []
        dup_srcfile = None
        for cand_fpath in dup_size_table[size]:
            is_dup = False
            if naive_dups:
                is_dup = (os.path.basename(finfo.path).lower()
                          == os.path.basename(cand_fpath).lower())
            else:
                is_dup = filecmp.cmp(finfo.path, cand_fpath)
            if is_dup:
                dup_srcfile = cand_fpath
                break
        if dup_srcfile:
            setattr(finfo, "is_dup", True)
            duplicates.append((finfo.path, dup_srcfile, size))
        else:
            setattr(finfo, "is_dup", False)
            dup_size_table[size].append(finfo.path)
    def print_duplicate_info(duplicates, list_duplicates):
        if len(duplicates):
            if list_duplicates:
                duplicate_strings = []
                for dup in duplicates:
                    prefix = os.path.commonprefix([dup[0], dup[1]])
                    dup1 = f"  1. {dup[0][len(prefix):]}"
                    dup2 = f"  2. {dup[1][len(prefix):]}"
                    duplicate_strings.append(f"{dup1}\n{dup2}")
                duplicate_list = "\n".join(duplicate_strings)
                logging.info(f"Found duplicates: {duplicate_list}")
            saved_str = human_redable_size(sum([dup[2] for dup in duplicates]))
            logging.info(f"Found {len(duplicates)} duplicates, saved {saved_str}")
    def handle_encoding(finfo, mts_encode_sh, encoded):
        fpath = finfo.path
        if mts_encode_sh and os.path.splitext(fpath)[1].lower() == '.mts':
            encoded_fdir = os.path.join(tmp_dir, str(len(encoded)))
            os.makedirs(encoded_fdir)
            encoded_fname = f"{os.path.basename(fpath)}.mp4"
            encoded_fpath = os.path.join(encoded_fdir, encoded_fname)
            args = ['sh', mts_encode_sh, fpath, encoded_fpath]
            encode_proc = subprocess.run(args, capture_output=True)
            if (encode_proc.returncode == 0):
                encoded.append((fpath,
                                os.stat(fpath).st_size,
                                os.stat(encoded_fpath).st_size))
                finfo.path = encoded_fpath
            else:
                logging.error(f"Failed to encode: {fpath}")
    def print_encoded_info(encoded, list_encoded):
        if len(encoded):
            if list_encoded:
                encoded_strings = []
                for enc in encoded:
                    sz0, sz1 = human_redable_size(enc[1]), human_redable_size(enc[2])
                    encoded_strings.append(f"{enc[0]}: {sz0} -> {sz1}")
                encoded_list = "\n  ".join(encoded_strings)
                logging.info(f"Encoded: {encoded_list}")
            total_sz0 = sum([enc[1] for enc in encoded])
            total_sz1 = sum([enc[2] for enc in encoded])
            saved_str = human_redable_size(total_sz0 - total_sz1)
            logging.info(f"Encoded {len(encoded)} files, saved {saved_str}")
    dup_size_table = {}
    duplicates = []
    encoded = []
    new_file_infos = []
    for fileinfo in progressbar(file_infos,
                                "Parsing and encoding media " if mts_encode_sh
                                else "Parsing media ",
                                40):
        new_file_info = copy.copy(fileinfo)
        handle_duplicate(new_file_info, duplicates, dup_size_table, naive_dups)
        if not new_file_info.is_dup:
            handle_encoding(new_file_info, mts_encode_sh, encoded)
            stat = os.stat(new_file_info.path)
            setattr(new_file_info, "size", stat.st_size)
            setattr(new_file_info, "date",
                    datetime.datetime.fromtimestamp(stat.st_mtime))
            new_file_infos.append(new_file_info)
    print_duplicate_info(duplicates, list_duplicates)
    print_encoded_info(encoded, list_encoded)
    return new_file_infos

def create_bucket(file_indices=[], size=0, date=None, split_index=None):
    return SimpleNamespace(file_indices=file_indices, size=size, date=date,
                           split_index=split_index)

def split_large_buckets(media_file_infos, buckets,
                        max_bucket_size=250*1.0e6, tolerance_perc=15):
    def create_split_buckets(file_indices, date):
        split_buckets = []
        split_buck = []
        split_buck_size = 0
        for file_index in file_indices:
            split_buck.append(file_index)
            split_buck_size += media_file_infos[file_index].size
            if split_buck_size > tol_max_bucket_size:
                split_buckets.append(
                    create_bucket(split_buck, split_buck_size, date,
                                  split_index=len(split_buckets)+1))
                split_buck = []
                split_buck_size = 0
        if split_buck_size:
            split_buckets.append(
                create_bucket(split_buck, split_buck_size, date,
                              split_index=len(split_buckets)+1))
        return split_buckets
    new_buckets = []
    large = 0
    tol_max_bucket_size = max_bucket_size * (100+tolerance_perc)/100.0
    for bucket in buckets:
        if bucket.size > tol_max_bucket_size:
            new_buckets.extend(
                        create_split_buckets(bucket.file_indices, bucket.date))
        else:
            new_buckets.append(bucket)
    logging.debug(f"Split large buckets: {len(buckets)} -> {len(new_buckets)}")
    return new_buckets

def generate_day_buckets(media_file_infos,
                         max_bucket_size=250*1.0e6):
    def canonical_day(date):
        return date.replace(hour=12, minute=0, second=0, microsecond=0, fold=0)
    # Init data-sorted file indices
    file_indices = list(range(len(media_file_infos)))
    date_sorted_file_indices = sorted(
                    file_indices, key = lambda index: media_file_infos[index].date)
    # Initial day table
    day_file_indicies = {}
    for index in date_sorted_file_indices:
        day = canonical_day(media_file_infos[index].date)
        if day not in day_file_indicies:
                day_file_indicies[day] = []
        day_file_indicies[day].append(index)
    # Join days into groups
    seq_days = sorted(day_file_indicies.keys())
    seq_days_sizes = [sum([media_file_infos[index].size
                          for index in day_file_indicies[day]])
                      for day in seq_days]
    day_seq_groups = []
    highest_joined_seq_index = -1
    for seq_i, day in enumerate(seq_days):
        if seq_i > highest_joined_seq_index:
            join_size = seq_days_sizes[seq_i]
            group_seq_indices = [seq_i]
            for join_seq_cand_i in range(seq_i + 1, len(seq_days)):
                new_join_size = join_size + seq_days_sizes[join_seq_cand_i]
                if (seq_days[join_seq_cand_i].year == seq_days[seq_i].year
                    and new_join_size < max_bucket_size):
                    group_seq_indices.append(join_seq_cand_i)
                else:
                    break
            day_seq_groups.append(group_seq_indices)
            highest_joined_seq_index = group_seq_indices[-1]
    logging.debug(f"Bucketed days: {len(seq_days)} -> {len(day_seq_groups)}")
    def join_file_indices(seq_day_indices):
        return sum([day_file_indicies for seq_day_index in seq_day_indices])
    # Generate buckets
    buckets = []
    for day_seq_group in day_seq_groups:
        days = [seq_days[index] for index in day_seq_group]
        group_file_indices = sum([day_file_indicies[day] for day in days], [])
        group_size = sum([seq_days_sizes[index] for index in day_seq_group])
        buckets.append(create_bucket(group_file_indices, group_size, days[-1]))
    return buckets

def print_tree(root, children_func=list, name_func=str):
    """Pretty print a tree, in the style of gnu tree"""
    space =  '    '
    branch = '│   '
    tee =    '├── '
    last =   '└── '
    # Inspired by https://stackoverflow.com/questions/9727673
    def tree(node, children_func, name_func, prefix: str = ''):
        """A recursive generator, given a tree
        will yield a visual tree structure line by line
        with each line prefixed by the same characters
        """
        contents = children_func(node)
        # contents each get pointers that are ├── with a final └── :
        pointers = [tee] * (len(contents) - 1) + [last]
        for pointer, path in zip(pointers, contents):
            yield prefix + str(pointer) + name_func(path)
            if len(children_func(path)):  # extend the prefix and recurse:
                extension = branch if pointer == tee else space
                # i.e. space because last, └── , above so no more |
                yield from tree(path, children_func, name_func,
                                prefix=prefix + extension)
    # Print the tree
    print(name_func(root))
    for line in tree(root, children_func, name_func):
        print(line)

def generate_output_fs_tree(media_file_infos, buckets):
    def get_date_quarter(date):
        return int(date.month / 4)
    def create_bucket_fs_entries(quarter_buckets, quarter_name):
        def create_bucket_name(bucket):
            def double_digit(number):
                return (f"0{number}" if number < 10 else f"{number}")
            root_name = f"{bucket.date.year}_{double_digit(bucket.date.month)}"
            day_name = double_digit(bucket.date.day)
            main_name = f"{root_name}_{day_name}"
            return (main_name if bucket.split_index is None
                    else f"{main_name}_{bucket.split_index}")
        return [(create_bucket_name(bucket),
                 set(sorted([media_file_infos[idx].path for idx in bucket.file_indices])))
                for bucket in quarter_buckets]
    # Map buckets to quarters
    Quarter = namedtuple('Quarter', ['year', 'index'])
    buckets_for_quarter = {}
    for bucket in buckets:
        quarter = Quarter(bucket.date.year,
                                         get_date_quarter(bucket.date))
        if quarter not in buckets_for_quarter:
            buckets_for_quarter[quarter] = []
        buckets_for_quarter[quarter].append(bucket)
    # Create structure as dicts
    fs_years = {}
    for quarter, quarter_buckets in buckets_for_quarter.items():
        if quarter.year not in fs_years:
            fs_years[quarter.year] = {}
        quarter_name = f"Q{quarter.index+1}"
        fs_entries = create_bucket_fs_entries(quarter_buckets, quarter_name)
        fs_years[quarter.year][quarter_name] = fs_entries
    logging.info(f"Found years: {sorted(fs_years.keys())}")
    # Transform to tuple-based tree structure
    fs_tree = []
    for year in sorted(fs_years.keys()):
        year_tree = []
        for quarter in sorted(fs_years[year].keys()):
            year_tree.append((quarter, fs_years[year][quarter]))
        fs_tree.append((year, year_tree))
    return fs_tree

def print_fs_tree(fs_tree, struct_only=True):
    full_children_lambda = lambda node: node[1] if type(node) == tuple else []
    def struct_only_lambda(node):
        # Use the fact that only the file list is given as a set
        content = full_children_lambda(node)
        return  [f"({len(content)} files)"] if type(content) == set else content
    print_tree(('', fs_tree),
               struct_only_lambda if struct_only else full_children_lambda,
               lambda node: str(node[0] if type(node) == tuple else node))

def write_output_fs_tree(fs_tree, zip_buckets=False,
                            flatten=False, struct_only=False, text_hints=True,
                            list_duplicates=False, mts_encode_sh=None,
                            list_encoded=False):
    def make_uniq_out_fname(srcfile, uniq_out_fnames):
        out_fname = os.path.basename(srcfile)
        if out_fname not in uniq_out_fnames:
            uniq_out_fnames[out_fname] = [srcfile]
        else:
            uniq_out_fnames[out_fname].append(srcfile)
            uniq_idx = len(uniq_out_fnames[out_fname])
            name, ext = os.path.splitext(out_fname)
            out_fname = f"{name}_{uniq_idx}{ext}"
        return out_fname
    fs_out_root = args.out_dir
    out_buckets = []
    for year_tuple in fs_tree:
        year_root = os.path.join(args.out_dir, str(year_tuple[0]))
        for quarter_tuple in year_tuple[1]:
            quarter_root = os.path.join(year_root, str(quarter_tuple[0]))
            for quarter_buckets in quarter_tuple[1]:
                bucket_root = os.path.join(quarter_root, str(quarter_buckets[0]))
                out_buckets.append((bucket_root, quarter_buckets[1]))
    for bucket_tuple in progressbar(out_buckets, "Generating output ", 40):
        bucket_dir = bucket_tuple[0]
        os.makedirs(bucket_dir)
        if not struct_only:
            uniq_out_fnames = {}
            bucket_hints = set()
            for srcfile in bucket_tuple[1]:
                out_fname = make_uniq_out_fname(srcfile, uniq_out_fnames)
                out_fpath = os.path.join(bucket_dir, out_fname)
                shutil.copy2(srcfile, out_fpath)
                dir_names = os.path.split(os.path.dirname(srcfile))
                if len(dir_names):
                    bucket_hints.add(dir_names[-1])
            def write_hints(hints, dir):
                for hint in hints:
                    open(f"{os.path.join(dir, hint)}.txt", 'a').close()
            write_hints(bucket_hints, bucket_dir)
            if zip_buckets:
                write_hints(bucket_hints, os.path.dirname(bucket_dir))
                shutil.make_archive(f"{bucket_dir}", 'zip', bucket_dir)
                shutil.rmtree(bucket_dir)

def main(args):
    logging.basicConfig(format='%(levelname)s: %(message)s',
                        level="DEBUG" if args.log_debug else "INFO")
    # Check pre-conditions
    dry_run = args.dry_run or args.dry_run_detailed
    if dry_run and (args.mts_encode_sh is not None):
        logging.info("Dry run will ignore encoding, "
                     "the actual run will probably have fewer buckets")
        args.mts_encode_sh = None
    # Parse files
    print(f"Parsing {args.directory} ...")
    media_file_infos, non_media_extensions = parse_fs_media_file_info_paths(args.directory, args)
    if len(non_media_extensions):
        logging.warning(f"Found these non-media extensions: {list(non_media_extensions)}")
    logging.info(f"Found {len(media_file_infos)} media files")
    if len(media_file_infos) == 0:
        return os.EX_OK
    # Fill file infos
    tmp_dir = os.path.join(args.out_dir, 'tmp')
    media_file_infos = prepare_media_file_infos(media_file_infos,
                                                args.naive_dups,
                                                args.list_dups,
                                                args.mts_encode_sh,
                                                tmp_dir,
                                                args.list_encoded)
    total_media_size = sum(file_info.size for file_info in media_file_infos)
    logging.info(f"Total media size is {human_redable_size(total_media_size)}")
    # Generate FS tree
    buckets = generate_day_buckets(media_file_infos)
    buckets = split_large_buckets(media_file_infos, buckets)
    out_fs_tree = generate_output_fs_tree(media_file_infos, buckets)
    if dry_run:
        # Handle dry run
        if not (args.dry_run_no_tree):
            print_fs_tree(out_fs_tree, not args.dry_run_detailed)
    else:
        # Generate the output FS tree
        write_output_fs_tree(out_fs_tree,
                             zip_buckets=args.zip,
                             struct_only=args.struct_only,
                             text_hints=not args.no_hints)
        shutil.rmtree(tmp_dir, ignore_errors=True)
    return os.EX_OK

parser = argparse.ArgumentParser()
parser.add_argument("directory")
parser.add_argument("--out-dir", default='./')
parser.add_argument("--zip", default=False, action="store_true")
parser.add_argument("--struct-only", default=False, action="store_true")
parser.add_argument("--no-hints", default=False, action="store_true")
parser.add_argument("--list-dups", default=False, action="store_true")
parser.add_argument("--naive-dups", default=False, action="store_true")
parser.add_argument("--no-images", default=False, action="store_true")
parser.add_argument("--no-videos", default=False, action="store_true")
parser.add_argument("--mts-encode-sh", default=None)
parser.add_argument("--list-encoded", default=False, action="store_true")
parser.add_argument("--log-debug", default=False, action="store_true")
parser.add_argument("--dry-run", default=False, action="store_true")
parser.add_argument("--dry-run-no-tree", default=False, action="store_true")
parser.add_argument("--dry-run-detailed", default=False, action="store_true")
args = parser.parse_args()
sys.exit(main(args))

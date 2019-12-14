# -*- coding: utf-8 -*-

import argparse
import os
import json

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument(
        "--old_stats",
        type=str,
        help="Path to the old stats file.")
    parser.add_argument(
        "--new_stats",
        type=str,
        help="Path to the new stats file.")

    flags, unparsed = parser.parse_known_args()

    old_stats_path = flags.old_stats
    new_stats_path = flags.new_stats

    print("--------------------------------------------------")
    arg_values = ""
    for arg in vars(flags):
        print(arg + "='" + str(getattr(flags, arg)) + "'")
    print("--------------------------------------------------")

    print("Loading old stats: " + str(old_stats_path))
    with open(old_stats_path) as old_stats_file:
        old_stats = json.load(old_stats_file)

    print("Loading new stats: " + str(new_stats_path))
    with open(new_stats_path) as new_stats_file:
        new_stats = json.load(new_stats_file)

    # if the new is better -> return 1
    old_f1 = float(old_stats['weighted avg']['f1-score'])
    new_f1 = float(new_stats['weighted avg']['f1-score'])

    old_C43_precision = float(old_stats['C43']['precision'])
    new_C43_precision = float(new_stats['C43']['precision'])
    old_C43_recall = float(old_stats['C43']['recall'])
    new_C43_recall = float(new_stats['C43']['recall'])

    print("f1: ", old_f1, new_f1)
    print("C43 precision: ", old_C43_precision, new_C43_precision)
    print("C43 recall: ", old_C43_recall, new_C43_recall)

    if old_f1 <= new_f1 and old_C43_precision <= new_C43_precision and old_C43_recall < new_C43_recall:
        print("yes")
        exit(1)

    print("no")
    exit(0)
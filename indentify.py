import argparse
import os
import sqlite3

def identify_sample(database, input):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Builds an audio fingerprint database from a folder of songs.",
        epilog="Example: python builddb.py -i songs-folder -o database.db"
    )
    parser.add_argument('-d', '--database', required=True, help="Database file")
    parser.add_argument('-i', '--input', required=True, help="Input sample")
    args = parser.parse_args()

    identify_sample(args.database, args.input)

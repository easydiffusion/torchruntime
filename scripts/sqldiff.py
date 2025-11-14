import sys
import sqlite3

if len(sys.argv) < 3:
    print("Usage: python sqldiff.py <file1> <file2>")
    exit()

QUERY = "SELECT * FROM pci_ids"
COL_NAME_QUERY = "PRAGMA table_info(pci_ids)"

db1 = sys.argv[1]
db2 = sys.argv[2]

conn1 = sqlite3.connect(db1)
conn2 = sqlite3.connect(db2)

cursor1 = conn1.cursor()
cursor2 = conn2.cursor()

res1 = set(cursor1.execute(QUERY).fetchall())
res2 = set(cursor2.execute(QUERY).fetchall())

cols = cursor1.execute(COL_NAME_QUERY).fetchall()
cols = tuple(col[1] for col in cols)

# Create dictionaries keyed by (vendor_id, device_id) for change detection
dict1 = {(row[0], row[2]): row for row in res1}
dict2 = {(row[0], row[2]): row for row in res2}

# Find changes where vendor_id and device_id match but other fields differ
changes_old = []
changes_new = []
for key in dict1.keys() & dict2.keys():
    old_row = dict1[key]
    new_row = dict2[key]
    # Check if device_name (index 3) or is_discrete (index 4) changed
    if old_row[1] != new_row[1] or old_row[3] != new_row[3] or old_row[4] != new_row[4]:
        changes_old.append(old_row)
        changes_new.append(new_row)

# Remove changed rows from additions/deletions
changed_keys = set(zip([row[0] for row in changes_old], [row[2] for row in changes_old]))
additions = {row for row in res2 - res1 if (row[0], row[2]) not in changed_keys}
deletions = {row for row in res1 - res2 if (row[0], row[2]) not in changed_keys}

if additions:
    print("### Additions")
    print("```")
    print(cols)
    for row in additions:
        print(row)
    print("```")

if deletions:
    print("### Deletions")
    print("```")
    print(cols)
    for row in deletions:
        print(row)
    print("```")

if changes_old:
    print("### Changes")
    print("#### Old values:")
    print("```")
    print(cols)
    for row in changes_old:
        print(row)
    print("```\n")
    print("#### Changed values:")
    print("```")
    print(cols)
    for row in changes_new:
        print(row)
    print("```")

import os, decimal
## Helper Functions ##

# Taken from https://www.cs.cmu.edu/~112/notes/notes-recursion-part2.html
# With modification to only return files with specified filetype
def listFiles(path, suffix = '.csv'):
    if os.path.isfile(path) and path.endswith(suffix):
        # Base Case: return a list of just this file
        return [ path ]
    elif os.path.isfile(path):
        return [ ]
    else:
        # Recursive Case: create a list of all the recursive results from
        # all the folders and files in this folder
        files = [ ]
        for filename in os.listdir(path):
            files += listFiles(path + '/' + filename)
        return files

# Adapted from course site:
# https://www.cs.cmu.edu/~112/notes/notes-2d-lists.html
# With modifications to return string value instead of printing to console
def stringFormat2dList(a):
    if (a == []):
        # So we don't crash accessing a[0]
        return str([])
    rows = len(a)
    cols = len(a[0])
    fieldWidth = maxItemLength(a)
    ret = "["
    for row in range(rows):
        if (row > 0):
            ret += "\n "
        ret += "[ "
        for col in range(cols):
            if (col > 0):
                ret += ", "
            # The next 2 lines print a[row][col] with the given fieldWidth
            formatSpec = "%" + str(fieldWidth) + "s"
            ret += formatSpec % str(a[row][col])
        ret += " ]"
    ret += "]"
    return ret

# Taken from course site: https://www.cs.cmu.edu/~112/notes/notes-2d-lists.html
# Helper function for print2dList.
# This finds the maximum length of the string
# representation of any item in the 2d list
def maxItemLength(a):
    maxLen = 0
    rows = len(a)
    cols = len(a[0])
    for row in range(rows):
        for col in range(cols):
            maxLen = max(maxLen, len(str(a[row][col])))
    return maxLen

# Copied from
# https://www.cs.cmu.edu/~112/notes/notes-variables-and-functions.html
def roundHalfUp(d):
    # Round to nearest with ties going away from zero.
    # You do not need to understand how this function works.
    rounding = decimal.ROUND_HALF_UP
    return int(decimal.Decimal(d).to_integral_value(rounding=rounding))
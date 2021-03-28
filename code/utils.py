
def shift_(freq):
    if freq == "Monthly":
        return 1
    elif freq == "Quarterly":
        return 4
    else:
        return None

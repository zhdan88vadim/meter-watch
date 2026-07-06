
def list_to_number(digits):
    """Convert list of digits to number"""
    valid_digits = [d for d in digits if d != -1]
    if not valid_digits:
        return -1
    return int("".join(map(str, valid_digits)))
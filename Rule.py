class Rule:
    def __init__(self, is_numeric, error_range):
        self.mIs_numeric = is_numeric
        self.mError_Values = error_range

    # Check a range of numeric or string values to check if anything has been violated
    def is_valid(self, value):
        if self.mIs_numeric == True:
            if value < self.mError_Values[0] or value > self.mError_Values[1]:
                return False
        else:
            for msg in self.mError_Values:
                if msg == value:
                    return False
        return True

    # Set a new range, used if user wants to repeatedly check a variable instead of a constant
    def set_range(self, new_range):
        self.mError_Values = new_range

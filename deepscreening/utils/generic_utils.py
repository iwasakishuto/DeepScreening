#coding: utf-8
def pad_string(string, max_len, padding="right"):
    method = {
        "right" : "ljust",
        "left"  : "rjust",
        "both"  : "center",
    }.get(padding, "ljust")
    return string.__getattribute__(method)(max_len, " ")

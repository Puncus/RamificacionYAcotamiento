def subscript(number):
    subscript_map = {
        "0": "₀",
        "1": "₁",
        "2": "₂",
        "3": "₃",
        "4": "₄",
        "5": "₅",
        "6": "₆",
        "7": "₇",
        "8": "₈",
        "9": "₉",
        "i": "ᵢ",
        }

    return ''.join(subscript_map[digit] for digit in str(number))

def gen_labels(amount):
    labels = []
    for i in range(amount):
        labels.append(f"x{subscript(f'{i+1}')}")
    return labels
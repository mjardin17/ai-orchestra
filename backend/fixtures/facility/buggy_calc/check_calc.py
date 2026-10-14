from calc import average


def test_average_of_three():
    assert average([1, 2, 3]) == 2


def test_average_of_one():
    assert average([5]) == 5

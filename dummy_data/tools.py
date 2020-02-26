import datetime
import random
from random import randrange


def random_date(start, l):
    current = start
    while l >= 0:
        curr = current + datetime.timedelta(minutes=randrange(60))
        yield curr
        l -= 1


def timestamps(l):
    startDate = datetime.datetime(2013, 9, 20, 13, 00)
    timestamps = []
    for x in random_date(startDate, l):
        timestamps.append(x.strftime("%d/%m/%y %H:%M"))
    return sorted(timestamps)


# random number generator
def random_gen(lower: int = 0, upper: int = 1):
    return random.randint(lower, upper)



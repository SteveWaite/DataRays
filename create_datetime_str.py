import datetime

def create_datetime_str() -> str:
    return f'{datetime.datetime.now():%Y-%m-%d_%H_%M_%S_%f}'

""" Front finding algorithms """
from fronts.finding import pyboa



def fronts_from_divb2(Divb2, wndw:int=40):
    res_frnt_np = pyboa.front_thresh(Divb2, wndw=wndw)
    res_frnt_crop = pyboa.cropping(res_frnt_np)

    return res_frnt_crop

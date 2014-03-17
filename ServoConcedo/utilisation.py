#!/usr/bin/python

# Standard Library Imports
import sys
import os
from glob import glob
for itertools import cycle

# C Dependency Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Custom Dependencies
sys.path.append('/home/nigel/python/OfferPandas/')
from OfferPandas import Frame, load_offerframe

sys.path.append('/home/nigel/python/pdtools/')
import pdtools

def get_plsr_stack(frame, filters, reserve_type="FIR", product_type="Both",
                   min_quantity=0.1):
    """
    Get the Reserve Stack for a given reserve configuration
    """

    # We don't want to mutate the original dictionary so we copy it.
    # Add in the product and reserve type filters
    fdict = filters.copy()
    fdict["Reserve_Type"] = reserve_type
    if product_type != "Both":
        fdict["Product_Type"] = product_type

    # Filter the dataset, ignore offers below the minimum quantity
    filtered_frame = frame.efilter(fdict)
    minquan_frame = filtered_frame.ge_mask("Quantity", min_quantity)

    # Return an Offer stack grouped by price ascending
    return minquan_frame.groupby(["Price"])["Quantity"].sum().cumsum()


def get_energy_stack(frame, filters, min_quantity=1.):
    """

    """
    # Filter the dataset, ignore offers below the minimum quantity
    filtered_frame = frame.efilter(filters)
    minquan_frame = filtered_frame.ge_mask("Quantity", min_quantity)

    # Return an Offer stack grouped by price ascending
    return minquan_frame.groupby(["Price"])["Quantity"].sum().cumsum()


def get_cumulative_capacity(frame, filters):
    """

    """
    filtered_frame = frame.efilter(filters)
    # Note spelling error
    return filtered_frame.groupby(["Station"])["Max_Ouput"].max()


def energy_reserve_matrix(energy, reserve, max_offer):
    """

    """
    combined = np.zeros((len(energy) + 1, len(reserve) + 1))

    # Add the energy offers
    combined[1:,:] += energy.reshape(len(energy), 1)

    # Add the reserve offers
    combined[:, 1:] += reserve.reshape(1, len(reserve))

    # Compare against the maximum usage constraint
    # Note, this won't strictly be accurate for mixed stations as it
    # uses a single combined capacity value.
    combined = np.where(combined <= max_offer, combined, max_offer)

    return combined


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


def asset_utilisation(energy, reserve, filters):

    # Need to remove these or the energy stack won't work
    reserve_type = filters.pop("Reserve_Type", "FIR")
    product_type = filters.pop("Product_Type", "PLSR")

    # Get the relevant values
    capacity = get_cumulative_capacity(energy, filters)
    estack = get_energy_stack(energy, filters)
    rstack = get_plsr_stack(reserve, filters, reserve_type=reserve_type, product_type=product_type)

    utilisation = energy_reserve_matrix(estack.values, rstack.values,
                                        capacity.sum())

    index = ["Energy"] + estack.index.values.tolist()
    cols = ["Reserve"] + rstack.index.values.tolist()

    # Return it as a DataFrame with prettier indexing and columns
    util_frame = pd.DataFrame(utilisation, index=index, columns=cols)

    return util_frame, capacity.sum()


def relative_utilisation(util, capacity):
    """

    """

    # Unpack the information
    index, columns, values = util.index, util.columns, util.copy().values

    # Update the values
    values[1:, 1:] = values[1:, 1:] / capacity * 100.

    # Repack into a DataFrame
    return pd.DataFrame(values, index=index, columns=columns)


def desired_utilisation(rel, energy, reserve):
    """

    """

    # Get the appropriate columns and indices
    columns = [x for x in rel.columns if x <= reserve and x != "Reserve"]
    index = [x for x in rel.index if x <= energy and x != "Energy"]

    # Wrap in try loop just in case zero offers etc are included
    try:
        return np.max(rel[columns].ix[index].values)
    except:
        return 0


def price_aggregation(prices, agg_func, *args, **kargs):
    # Get datetime objects if they aren't already
    if prices["Trading_Date"].dtype != np.datetime64:
        prices["Trading_Date"] = pd.to_datetime(prices["Trading_Date"])

    # Get the Day of the week
    prices["Weekdays"] = prices["Trading_Date"].apply(weekday_weekend)

    # Apply the Year Month
    prices["Year_Month"] = prices["Trading_Date"].apply(year_month)

    # Groupby and return values
    return prices.groupby(["Year_Month", "Weekdays",
                            "Trading_Period"])["Price SUM"].aggregate(
                            agg_func, *args, **kargs)


def weekday_weekend(x):
    return "Weekday" if x.weekday() <= 4 else "Weekend"


def year_month(x):
    return x.strftime('%Y_%m')
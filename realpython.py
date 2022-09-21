from array import array
import math
import statistics
import numpy as np
import scipy.stats
import pandas as pd

x = [8.0, 1, 2.5, 4, 28.0]
x_with_nan = [8.0, 1, 2.5, math.nan, 4, 28.0]
x
x_with_nan

math.isnan(np.nan), np.isnan(math.nan)

y, y_with_nan = np.array(x), np.array(x_with_nan)
z, z_with_nan = pd.Series(x), pd.Series(x_with_nan)
y

y_with_nan
z_with_nan

mean_ = sum(x) / len(x)
mean_

mean_ = statistics.mean(x)
mean_

mean_ = statistics.mean(x_with_nan)
mean_

mean_ = statistics.fmean(x_with_nan)
mean_

mean_ = np.mean(y)
mean_

np.mean(y_with_nan)
np.nanmean(y_with_nan)

mean_ = z.mean()
mean_

z_with_nan.mean()

0.2 * 2 + 0.5 * 4 + 0.3 * 8

x = [8.0, 1, 2.5, 4, 28.0]
w = [0.1, 0.2, 0.3, 0.25, 0.15]

wmean = sum(w[i] * x[i] for i in range(len(x))) / sum(w)

wmean = sum(x_ * w_ for (x_, w_) in zip(x, w)) / sum(w)

y, z, w = np.array(x), pd.Series(x), np.array(w)

wmean = np.average(y, weights=w)

wmean
(w * y).sum() / w.sum()

w = np.array([0.1, 0.2, 0.3, 0.0, 0.2, 0.1])
(w * y_with_nan).sum() / w.sum()

np.average(y_with_nan, weights=w)
np.average(z_with_nan, weights=w)

hmean = len(x) / sum(1 / item for item in x)

hmean = statistics.harmonic_mean(x)
hmean

statistics.harmonic_mean(x_with_nan)

statistics.harmonic_mean([1, 0, 2])
# statistics.harmonic_mean([1, 2, -2]) # Raises StatisticsError

scipy.stats.hmean(y)

gmean = 1

for item in x:
        gmean *= item

gmean **= 1 / len(x)

gmean = statistics.geometric_mean(x)
gmean

gmean = statistics.geometric_mean(x_with_nan)
gmean

scipy.stats.gmean(y)
scipy.stats.gmean(z)

n = len(x)
if n % 2:
    median_ = sorted(x)[round(0.5*(n-1))]
else:
    x_ord, index = sorted(x), round(0.5 * n)
   
    ### median_ = 0.5 * (x_ord[index-1] + x_ord[index]) #### i am generating an error with this code

median_ = statistics.median(x)
median_

median_ = statistics.median(x[:-1])
median_

statistics.median_low(x[:-1])
statistics.median_high(x[:-1])

statistics.median(x_with_nan)
statistics.median_low(x_with_nan)

statistics.median_high(x_with_nan)

median_ = np.median(y)
median_

median_ = np.median(y[:-1])
median_

np.nanmedian(y_with_nan)
np.nanmedian(y_with_nan[:-1])

z.median()
z_with_nan.median()

u = [2, 3, 2, 8, 12]

mode_ = max((u.count(item), item) for item in set(u))[1]
mode_

mode_ = statistics.mode(u)
mode_

mode_ = statistics.multimode(u)
mode_

v = [12, 15, 12, 15, 21, 15, 12]
##statistics.mode(v)  # Raises StatisticsError
statistics.multimode(v)

statistics.mode([2, math.nan, 2])

statistics.multimode([2, math.nan, 2])
statistics.mode([2, math.nan, 0, math.nan, 5])
statistics.multimode([2, math.nan, 0, math.nan, 5])

u, v = np.array(u), np.array(v)
mode_ = scipy.stats.mode(u)
mode_

mode_ = scipy.stats.mode(v)
mode_

mode_.mode
mode_.count

u, v, w = pd.Series(u), pd.Series(v), pd.Series([2, 2, math.nan])
u.mode()




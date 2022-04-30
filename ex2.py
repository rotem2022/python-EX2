
import pandas as pd

import numpy as np
from numpy.ma.core import append
import matplotlib.pyplot as plt

# numpy:
# Q 1.1
import numpy as np

def half(x):
  z=np.array(x[1::2,::2])
  z= z * 0.5
  return z

# Q 1.2
from numpy.core.fromnumeric import reshape
import numpy as np

def outer_product(x, y):
  z = x[:,None]*y
  return z

#Q 1.3
import numpy as np
def extract_logical(x, arr):
  ind = (arr % 1) == 0
  z = x[ind == True]
  return z , ind



def extract_integer(x, arr):
    z, _ = extract_logical(x, arr)
    ind = np.zeros((x.ndim, z.size), dtype=int)
    return z, ind

#Q 1.4 (1)
def norm(x):
  return sum(x**2) ** 0.5


def calc_norm(x, axis=0):
    res = np.apply_along_axis(norm, not axis, arr=x)
    return res

#(2)
def normalize(x, axis=0):
  normals = calc_norm(x, axis)
  if axis == 0:
    x = x.transpose()
  res = x / normals[None , :]
  if axis ==0:
    res = res.transpose()
  return res


a=np.arange(16).reshape(4,4)
print(a)
print(calc_norm(a, 1), '\n')
print(normalize(a, 1), '\n')


# Q 1.5

def matrix_norm(x, k=1000):
    A1 = np.random.randint(0, 2, (k, x.shape[0]))
    A2 = normalize(A1, 1)
    z = A2 @ x
    u = calc_norm(z, 1)

    return u.max()


# Q 1.6
import numpy as np


def det(A):
    res = 0
    if A.shape == (2, 2):
        return A[0][0] * A[1][1] - A[0][1] * A[1][0]

    for i in range(A.shape[0]):
        tmp = np.delete(A, i, 0)
        tmp = np.delete(tmp, 0, 1)
        res += det(tmp) * ((-1) ** (i + 1)) * (A[i, 0])
    return res


# Q 1.7


im = plt.imread('image.png')


def segment(im, thresh=128):
    res = np.copy(im)
    if im.ndim == 2:
        res[im < thresh] = 0
        res[im >= thresh] = 255
        return res

    if im.ndim == 3:
        im_bw = np.mean(im, axis=2)
        return segment(im_bw, thresh)

# Q1.8 ???


#Q 1.9

def is_magic(a):
  row_sum = np.sum(a, axis =1)
  col_sum  = np.sum(a, axis =0)
  diag_sum = a.trace()
  if np.array_equal(row_sum, col_sum) and np.all(row_sum == diag_sum) and a[::-1].trace() == diag_sum:
    return True
  return False



#pandas:
# Q2.1


def x3_plus_1(s):
    res = s.copy()
    res = res.mask(res % 2 == 0, res / 2)
    res = res.mask(res % 2 == 1, (3 * res) + 1)
    return res


# Q2.2
def right_str(str):
    if str[0].isupper():
        return str.upper()
    return str.lower()


def reindex_up_down(s):
    ind = pd.Series(s.index)
    tmp = ind.apply(right_str)
    return (pd.Series(s.values, index=tmp))


# Q2.3
def no_nans_idx(s):
    list = np.array(s.tolist())
    res = (list == list.astype(float))
    return pd.Series(res, index=s.index)


# Q2.4

def partial_sum(s):
    tmp = np.array(no_nans_idx(s))
    tmp2 = np.array(s)
    res = tmp2[tmp]
    return res.sum() ** 0.5


# Q2.5
def partial_eq(s1, s2):
    tmp1 = s1.dropna()
    tmp2 = s2.dropna()
    return tmp1 == tmp2


# s1 = pd.Series([1,3,np.nan,5.2], index=['aGJ', 'Bb', 'c','d'])
# s2 = pd.Series([1,2,np.nan, 4], index=['a', 'b', 'c','d'])
# print(reindex_up_down(s1))


# Q2.6
def dropna_mta_style(df, how='any'):

    a = pd.notna(df[:])
    b = df.T
    b = pd.notna(b[:])
    if how ==' any':
        delte_rows = df[(a == True).all(1)]
        delete_cols = (df.T[(b == True).all(1)])
    else:
        delte_rows = df[(a == True).any(1)]
        delete_cols = (df.T[(b == True).any(1)])

    rows = set(df.index.tolist())
    rows_to_delete = set(delte_rows.index)
    rows_to_delete = rows.difference(rows_to_delete)

    cols = set(df.columns)
    cols_to_delete = set(delete_cols.index)
    cols_to_delete = cols.difference(cols_to_delete)

    res = df.drop(cols_to_delete, axis='columns')
    res = res.drop(rows_to_delete, axis='rows')
    print(res)



df = pd.DataFrame(np.array([[1, 2, 14, 4, 5], [6, 8, 12, 7, np.nan], [9, np.nan, 13, 10, np.nan]]))
df2 = pd.DataFrame(np.array([[np.nan, 2, 3, 4, 5], [1, 8, np.nan, 4, 5], [1, 9, np.nan, 4, 5]]))

dropna_mta_style(df)

#   Q 2.7

def get_n_largest(df, n=0, how='col'):
  print(df)
  if how=='row':
    df=df.T
  print(df)
  res_df = df.apply(lambda x: x.sort_values(ascending=False).values)
  return res_df.loc[[n]]

# Q 2.8

def unique_dict(df, how='cols'):
    print('shape:', df.shape)
    ind = 1
    lst = list(df.columns)
    print(df)
    if how == 'rows':
        print(df)
        ind = 0
        lst = list(df.index)

    my_dict = dict()
    for i in range(0, df.shape[ind]):
        if how == 'cols':
            my_dict[lst[i]] = (df[(lst[i])].values)
        else:
            my_dict[lst[i]] = (df.loc[[i]].values)
    return my_dict



#Q 2.9
def upper(df):
    return df.apply(lambda row: toupper(row))

def toupper(row):
    return row.apply(lambda x: x.upper() if x == str(x) else x)


df = pd.DataFrame(np.array([[1, 'xx', 7, 'yy', 3],
                            [1, np.nan, np.nan, 4, '1fafa'],
                            [1, np.nan, np.nan, 'sfasva2', 5]]))

print(upper(df))
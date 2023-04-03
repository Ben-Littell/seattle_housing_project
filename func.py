import numpy as np
import matplotlib.pyplot as plt
import math
import random as rd
import pandas as pd

plt.style.use('seaborn')


def mean(data):
    total = sum(data)
    m = total / len(data)
    return m


def median(data):
    data.sort()
    if len(data) % 2 == 0:
        m = (data[len(data) // 2] + data[len(data) // 2 - 1]) / 2
    else:
        m = data[len(data) // 2]
    return m


def variance(data):
    new_list = [(val - mean(data)) ** 2 for val in data]
    v = mean(new_list)
    return v


def stand_dev(data):
    v = variance(data)
    s = math.sqrt(v)
    return s


def elem_stats(data):
    new_dict = {
        'mean': mean(data),
        'median': median(data),
        'variance': variance(data),
        'std': stand_dev(data),
        'min': min(data),
        'max': max(data)
    }
    return new_dict


def get_random(df, per):
    num_per = int(per/100 * len(df))
    training = df.sample(num_per)
    training = training.sort_index()
    test = df.drop(training.index)
    return training, test


def regression_eqn(ind_array, dep_array, linear=True):
    # input as two arrays or 2 columns of a DF
    x_4 = (ind_array**4).sum()
    x_3 = (ind_array**3).sum()
    x_2 = (ind_array**2).sum()
    x_1 = (ind_array).sum()
    n = len(ind_array)
    xy_2 = ((ind_array**2 * dep_array)).sum()
    xy = (ind_array * dep_array).sum()
    if linear:
        matrix1 = [[x_2, ind_array.sum()], [ind_array.sum(), n]]
        matrix2 = [[xy], [dep_array.sum()]]
        invarray1 = np.linalg.inv(matrix1)
        solution = np.dot(invarray1, matrix2)
        return solution
    else:
        matrix1 = [[x_4, x_3, x_2], [x_3, x_2, x_1], [x_2, x_1, n]]
        matrix2 = [[xy_2], [xy], [dep_array.sum()]]
        invarray1 = np.linalg.inv(matrix1)
        solution = np.dot(invarray1, matrix2)
        return solution[0][0], solution[1][0], solution[2][0]


def sigma_xy(xd, yd):
    nlist = []
    for i in range(len(xd)):
        nlist.append((xd[i] * yd[i]))
    return sum(nlist)


def least_sqrs(xd, yd):
    matrix1 = [[sum(val ** 2 for val in xd), sum(xd)], [sum(xd), len(xd)]]
    matrix2 = [sigma_xy(xd, yd), sum(yd)]
    array1 = np.array(matrix1)
    array2 = np.array(matrix2)
    invarray1 = np.linalg.inv(array1)
    solution = np.dot(invarray1, array2)
    return solution


def residuals(xd, yd, n=2):
    xdl = xd.tolist()
    ydl = yd.tolist()
    coeff2, coeff1, y_int = regression_eqn(xd, yd, linear=False)
    ys = [(coeff2 * (val**2)) + (coeff1*val) + y_int for val in xdl]
    r = [yd[n]-ys[n] for n in range(len(ydl))]
    mr = mean(r)
    stdr = stand_dev(r)
    return r, mr, stdr


def scatter_plot_er_2(data1, data2, coeff2, coeff1, y_int, std, title='Graph', xt='X', yt='Y', n=2):
    data1 = data1.tolist()
    data2 = data2.tolist()
    y_vals = []
    e1 = []
    e2 = []
    x_data = [min(data1), max(data1)]
    for val in range(len(data1)):
        ans = (coeff2 * (data1[val]**2)) + (coeff1*data1[val]) + y_int
        y_vals.append(ans)
    for val in range(len(data1)):
        ans = (coeff2 * (data1[val]**2)) + (coeff1*data1[val]) + y_int +(n*std)
        e1.append(ans)
    for val in range(len(data1)):
        ans = (coeff2 * (data1[val]**2)) + (coeff1*data1[val]) + y_int -(n*std)
        e2.append(ans)
    plt.plot(data1, y_vals, '-r')
    plt.plot(data1, e1, '--r')
    plt.plot(data1, e2, '--r')
    plt.scatter(data1, data2)
    plt.title(title)
    plt.xlabel(xt)
    plt.ylabel(yt)
    plt.text(x_data[1], y_vals[1], f'Y={round(coeff2, 5)}*X^2+{round(coeff1, 2)}X+{round(y_int, 2)}', color='g')
    plt.show()


def scatter_plot(data1, data2, slope, y_int, xt='X', yt='Y', title='Graph'):
    y_vals = []
    x_data = [min(data1), max(data1)]
    for val in range(2):
        ans = (slope * x_data[val]) + y_int
        y_vals.append(ans)
    plt.plot(x_data, y_vals, '-r')
    plt.scatter(data1, data2)
    plt.title(title)
    plt.xlabel(xt)
    plt.ylabel(yt)
    plt.text(x_data[1], y_vals[1], f'Y={round(slope, 4)}*X+{round(y_int, 4)}', color='g')
    plt.show()


def scatter_plot_bilin(data1, data2, coeff2, coeff1, y_int, title='Graph', xt='X', yt='Y'):
    data1 = data1.tolist()
    data1_s = sorted(data1)
    data2 = data2.tolist()
    y_vals = []
    e1 = []
    e2 = []
    x_data = [min(data1), max(data1)]
    for val in range(len(data1)):
        ans = (coeff2 * (data1_s[val]**2)) + (coeff1*data1_s[val]) + y_int
        y_vals.append(ans)
    plt.plot(data1_s, y_vals, '-r')
    plt.scatter(data1, data2)
    plt.title(title)
    plt.xlabel(xt)
    plt.ylabel(yt)
    plt.text(x_data[1], y_vals[1], f'Y={round(coeff2, 5)}*X^2+{round(coeff1, 2)}X+{round(y_int, 2)}', color='g')
    plt.show()


def cal_stats_column(df, v_dict, c_list):
    stats_dict = {}
    for key in v_dict:
        stats_dict[key] = {}
        ins = v_dict[key]
        t = df.loc[ins]
        for val in c_list:
            tl = t[val].values.tolist()
            stats_dict[key][val] = elem_stats(tl)
    return stats_dict


def search_engine(df):
    # t_zip = 'WA 98001'
    # t_bed = 3
    # t_bath = 2
    # t_floors = 2
    # t_living = 1700
    # t_price = 200000
    t_zip = input('Enter your desired state zipcode: ')
    t_bed = int(input('Enter your desired number of bedrooms: '))
    t_bath = int(input('Enter your desired number of bathrooms: '))
    t_floors = int(input('Enter your desired number of floors: '))
    t_living = int(input('Enter your desired amount of living space: '))
    t_price = int(input('Enter your desired price: '))
    df.loc[len(df.index)] = [t_bed, t_bath, t_floors, t_living, t_price, t_zip]

    norm_df = pd.DataFrame()
    norm_df['bedrooms'] = (df.bedrooms - df.bedrooms.mean()) / df.bedrooms.std()
    norm_df['bathrooms'] = (df.bathrooms - df.bathrooms.mean()) / df.bathrooms.std()
    norm_df['floors'] = (df.floors - df.floors.mean()) / df.floors.std()
    norm_df['sqft_living'] = (df.sqft_living - df.sqft_living.mean()) / df.sqft_living.std()
    norm_df['price'] = (df.price - df.price.mean()) / df.price.std()
    norm_df['statezip'] = df.statezip

    norm_user_vals = norm_df.iloc[-1]  # norm_user_vals['bedrooms'] gives the normalized value for bedrooms
    norm_df.drop(index=len(df.index)-1, inplace=True)

    norm_df['distance'] = ((norm_df['bedrooms'] - norm_user_vals['bedrooms'])**2 +
                           (norm_df['bathrooms'] - norm_user_vals['bathrooms'])**2 +
                           (norm_df['floors'] - norm_user_vals['floors'])**2 +
                           (norm_df['sqft_living'] - norm_user_vals['sqft_living'])**2 +
                           (norm_df['price'] - norm_user_vals['price'])**2)
    zips = norm_df[norm_df['statezip'] == t_zip]
    zips2 = zips['distance'].sort_values()
    zips3 = zips2.head()
    zips4 = zips3.index.values.tolist()
    print(df.loc[zips4])





import numpy as np
import datetime
import copy
def print_log(data=None, level='INFO'):
    print("[%s] [%s] %s" % (datetime.datetime.now().strftime(
        "%Y/%m/%d %H:%M:%S"), level, data))

def cal_relative_diff_np(real_data, expect_data, diff_thd):
    a = np.abs(np.subtract(real_data, expect_data))
    b1 = np.maximum(np.abs(real_data), (np.abs(expect_data)))
    b2 = float((1.0 / (1 << 14)) / diff_thd)
    b = np.add(np.maximum(b1, b2), 10e-10)
    result = np.where(a < diff_thd, a, a / b)
    return result

def cal_nan_inf_diff(real_data, expect_data, output_dtype, diff_abs, diff_thd):
    err_diff = []
    err_idx = []
    if output_dtype == 'fp32':
        inf_value = 3.4028e38
    elif output_dtype == 'bf16':
        inf_value = 3.38e38
    else:
        inf_value = 65504
    real_data_copy = copy.deepcopy(real_data)
    expect_data_copy = copy.deepcopy(expect_data)
    inf_idx = np.where(np.isinf(real_data_copy))[0]
    pos_inf_idx = np.where(real_data_copy[np.isinf(real_data_copy)] > 0)[0]
    neg_inf_idx = np.where(real_data_copy[np.isinf(real_data_copy)] < 0)[0]
    real_data_copy[inf_idx[pos_inf_idx]] = inf_value
    real_data_copy[inf_idx[neg_inf_idx]] = -inf_value
    inf_idx = np.where(np.isinf(expect_data_copy))[0]
    pos_inf_idx = np.where(expect_data_copy[np.isinf(expect_data_copy)] > 0)[0]
    neg_inf_idx = np.where(expect_data_copy[np.isinf(expect_data_copy)] < 0)[0]
    expect_data_copy[inf_idx[pos_inf_idx]] = inf_value
    expect_data_copy[inf_idx[neg_inf_idx]] = -inf_value

    num_idx = np.where(~np.isnan(real_data_copy) & ~np.isinf(real_data_copy) & ~np.isnan(expect_data_copy) & ~np.isinf(
        expect_data_copy))
    nan_inf_idx = np.setdiff1d(np.arange(len(real_data_copy)), num_idx)
    rdiff = cal_relative_diff_np(real_data_copy[num_idx].astype(np.float32),
                                 expect_data_copy[num_idx].astype(np.float32),
                                 diff_thd)
    num_err_diff = rdiff[rdiff > diff_thd]
    diff_idx_list = num_idx[0]
    num_err_idx = diff_idx_list[np.where(rdiff > diff_thd)]

    real_data_str = list(map(lambda x: str(x), real_data[nan_inf_idx].tolist()))
    expect_data_str = list(map(lambda x: str(x), expect_data[nan_inf_idx].tolist()))
    temp_err_idx = np.where(np.array(real_data_str) != np.array(expect_data_str))[0]
    nan_inf_err_idx = nan_inf_idx[temp_err_idx]
    nan_inf_err_diff = diff_abs[nan_inf_err_idx]

    err_idx = num_err_idx.tolist() + nan_inf_err_idx.tolist()
    err_diff = num_err_diff.tolist() + nan_inf_err_diff.tolist()

    return np.array(err_diff), np.array(err_idx)

def cal_relative_diff(real_data, expect_data, diff_thd, type_str='fp16'):
    if 'nan' in str(expect_data) or 'inf' in str(expect_data):
        if type_str.lower() == 'fp16':
            expect_data = 65504
        else:
            expect_data = 3.4028e38
    diff = abs(float(real_data) - float(expect_data)) 
    if abs(float(real_data) - float(expect_data)) < diff_thd:
        result = diff
    else:
        result = diff / (float(max(abs(real_data), abs(expect_data))) + 10e-10)
    return result

def display_error_output(real_data, expect_data, err_idx, relative_diff):
    print_log(
        'Error Line-----------------------------------------------------------------------------')
    print_log('Loop \t ExpectOut \t RealOut \t FpDiff \t RateDiff')
    print_log(
        '---------------------------------------------------------------------------------------')
    count = 0
    len_err = len(err_idx)        
    for i in err_idx:
        count += 1
        if count < 10 or (90 < count < 100):
            print_log('%08d \t %.7f \t %.7f \t %.7f \t %.7f' % (
                i, expect_data[i], real_data[i], abs(np.float64(
                    expect_data[i]) - np.float64(real_data[i])),
                relative_diff[count - 1]))
        elif count == 10 or (count == 100 and len_err > 100):
            dot_3 = '...'
            print_log('%08s \t %07s \t %07s \t %07s \t %07s' % 
                      (dot_3, dot_3, dot_3, dot_3, dot_3))
        elif count > 100:
            break

    print_log(
        'Max-RE line:---------------------------------------------------------------------------')
    max_error = max(relative_diff)
    m_idx_list = err_idx[np.where(relative_diff == max_error)]
    m_count = 0
    for m_idx in m_idx_list:
        m_count += 1
        if m_count < 4:
            print_log('%08d \t %.7f \t %.7f \t %.7f \t %.7f' % (
                m_idx, expect_data[m_idx], real_data[m_idx],
                abs(np.float64(expect_data[m_idx]) - 
                    np.float64(real_data[m_idx])),
                max_error))
        else:
            break
    print_log(
        '---------------------------------------------------------------------------------------')

def display_output(real_data, expect_data, start, end, diff_thd, expect_fp32_data=None):
    def display_inner(idx):
        j = idx + start
        diff_rate = cal_relative_diff(
            expect_data[j], real_data[j], diff_thd)
        if "inf" in str(expect_data[j]) or "nan" in str(expect_data[j]):
            diff_abs = "inf" if "inf" in str(expect_data[j]) else "nan"
            if expect_fp32_data is not None:
                print_log('%08d \t %-7s \t %-7s \t %-7s \t %-7s \t %-7s' % (
                    start + idx + 1, expect_fp32_data[j], expect_data[j], real_data[j], diff_abs, diff_rate))
            else:
                print_log('%08d \t %-7s \t %-7s \t %-7s \t %-7s' % (
                    start + idx + 1, expect_data[j], real_data[j], diff_abs, diff_rate))
        else:
            diff_abs = abs(np.float64(
                expect_data[j]) - np.float64(real_data[j]))
            if expect_fp32_data is not None:
                print_log('%08d \t %0.7f \t %0.7f \t %0.7f \t %0.7f \t %0.7f' % (
                    start + idx + 1, expect_fp32_data[j], expect_data[j], real_data[j], diff_abs, diff_rate))
            else:
                print_log('%08d \t %0.7f \t %0.7f \t %0.7f \t %0.7f' % (
                    start + idx + 1, expect_data[j], real_data[j], diff_abs, diff_rate))
    print_log(
        '---------------------------------------------------------------------------------------')
    if expect_fp32_data is not None:
        print_log(
            'Loop \t ExpFP32Out \t ExpFP16Out \t NPUOut \tFpDiff(min) \t RateDiff')
    else:
        print_log('Loop \t ExpectOut \t RealOut \t FpDiff \t RateDiff')
    print_log(
        '---------------------------------------------------------------------------------------')
    split_count = int(end - start)
    if split_count <= 20:
        for i in range(split_count + 1):
            display_inner(i)
    else:
        for i in range(10):
            display_inner(i)
        print_log('...   \t   ...   \t   ...   \t   ...   \t   ...')
        for i in range(split_count - 10 + 1, split_count + 1):
            display_inner(i)

def data_compare_np(npu_output, cpu_output, diff_thd=0.01, pct_thd=0.05, max_diff_hd=0.1,
                    rtol=0.005, atol=0.000025, output_dtype=None):
    max_error_idx = 10000000
    real_data = npu_output.flatten()
    data_compe = cpu_output.flatten()
    if real_data.size == 0 and real_data.size == data_compe.size:
        print_log(
            'The npu_output is [],and it is same as bm_output, the result of data_compare is \"PASS\"')
        return "PASS", 100.0, 0
    start = 0
    end = real_data.size - 1
    if end < start:
        end = start
    max_error = 0
    result = "FAIL"
    if real_data.size != data_compe.size:
        print_log(
            'Error, the size of npu output[%s] and benchmark[%s] is not equal.' % (real_data.size, data_compe.size))
        return result, 0.0, max_error
    overflows_count = data_compe[np.isinf(
        data_compe)].size + data_compe[np.isnan(data_compe)].size
    if overflows_count > 0:
        print_log('Overflow,size:%s,benchmark_output:%s, %s' % (
            overflows_count, data_compe[np.isinf(data_compe)][0:10], data_compe[np.isnan(data_compe)][0:10]))
    split_count = int(end - start + 1) if end != start else 1
    print_log('split_count:%s; max_diff_hd:%s;' %
              (float(split_count), max_diff_hd))
    has_nan_inf = False
    if 'nan' in str(real_data) or 'inf' in str(real_data) or 'nan' in str(data_compe) or 'inf' in str(data_compe):
        has_nan_inf = True
    try:
        diff_abs = np.abs(np.subtract(real_data.astype(
            np.float32), data_compe.astype(np.float32)))      
    except MemoryError:
        return result, 0.0, max_error
    if has_nan_inf:
        err_diff, err_idx = cal_nan_inf_diff(real_data, data_compe, output_dtype, diff_abs, diff_thd)
    else:
        diff_index = np.where(diff_abs > 0)
        rdiff = cal_relative_diff_np(real_data[diff_index].astype(np.float32),
                                        data_compe[diff_index].astype(np.float32),
                                        diff_thd)
        err_diff = rdiff[rdiff > diff_thd]
        diff_idx_list = diff_index[0]
        err_idx = diff_idx_list[np.where(rdiff > diff_thd)]
    fulfill_percent = float(split_count - err_diff.size) / \
                        float(split_count) * 100.0
    display_output(real_data, data_compe, start, end, diff_thd)
    pct_thd = (1 - pct_thd) * 100.0
    result = "PASS" if (fulfill_percent >= pct_thd) else "FAIL"
    if len(err_diff) > 0:
        max_error = max(err_diff[0:max_error_idx])
        if max_error > max_diff_hd:
            result = "FAIL"
            print("max_error >= max_diff_hd")
    print_log(
        '---------------------------------------------------------------------------------------')
    print_log('DiffThd  \t PctThd   \t PctRlt   \t Result')
    print_log(
        '---------------------------------------------------------------------------------------')
    print_log('%.4f     \t %.2f%%   \t %.6f%%   \t %s' %
                (diff_thd, pct_thd, fulfill_percent, result))
    if len(err_diff) > 0:
        print_log('Max-RelativeError is: %s. Threshold is: %s.' %
                    (max_error, max_diff_hd))
    if result == "FAIL":
        display_error_output(real_data, data_compe,
                                err_idx, err_diff[0:max_error_idx])
    return result, fulfill_percent
    


    
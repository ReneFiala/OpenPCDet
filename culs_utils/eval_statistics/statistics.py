import pandas as pd
import scipy.stats as stats
from pathlib import Path
import os
import numpy as np
import openpyxl as xl
from joblib import Parallel, delayed

ROOT = Path("./metrics_2026-09-20_grouped_voxelrcnn")
METHOD_NAMES = ['rnd_0', 'rnd_1', 'rnd_2', 'rnd_3',
           'sp_ag', 'sp_ag_num-c', 'sp_ag_num-w',
           'ci-3b', 'ci-3b-qrt', 'ci-5b',
           'ci-3b_sp', 'ci-3b-qrt_sp', 'ci-5b_sp',
           'lai-3b', 'lai-5b', 'lai-3b_sp', 'lai-5b_sp',
           'pv-3b', 'pv-3b_sp', 'pv-5b', 'pv-5b_sp']
           # 'kmeans_all']
MINIMIZE_METRICS = ["mae", "rmse", "amae", "armse"]
PRINT_RESULTS = False
TESTS_ALPHA = 0.01 # hladina významnosti
OMEGA_LOWER_CI_LIMIT = 0.14

#%% Load data into dict{ method_name: list[list[float]] }

def get_method_names(root: Path) -> list[str]:
    dirs = os.listdir(root)
    methods = set([x[:-6] for x in dirs])
    times = [os.path.getmtime(root / f"{x}-t0-v1" / "log_test.csv") for x in methods]
    # return sorted(methods)
    return [x for _, x in sorted(zip(times, methods))]

def load_method(name: str, metric: str) -> list:
    output = []
    for t in range(5):
        suboutput = []
        for v in range(5):
            if t == v:
                continue
            df = pd.read_csv(ROOT / f"{name}-t{t}-v{v}" / "log_test.csv", sep=",")
            if metric not in df.columns:
                df = pd.read_csv(ROOT / f"{name}-t{t}-v{v}" / "aver_counts.csv", sep=";")
            if metric not in df.columns:
                df = pd.read_csv(ROOT / f"{name}-t{t}-v{v}" / "real_ap.csv", sep=",")
            suboutput.append(df[metric][0])
        output.append(suboutput)
    return output

def load_all_methods(methods: list[str], metric: str) -> dict:
    metrics = [load_method(x, metric) for x in methods]
    return {k: np.asarray(v) for k, v in zip(methods, metrics)}

#%% Utilities

GREEN = "\033[32m"
RED = "\033[31m"
RESET = "\033[0m"

def flatten_list(xss):
    return [x for xs in xss for x in xs]

def do_tests(test, data, flatten=False, unpack=False, kw={}, attr=None):
    key_length = get_key_length(data)
    results = []
    for k, v in data.items():
        if flatten:
            v = flatten_list(v)
        if unpack:
            result = test(*v, **kw)
        else:
            result = test(v, **kw)
        if attr:
            result = result._asdict()[attr]
        results.append(result)
        if PRINT_RESULTS:
            if attr == "pvalue":
                print_pvalue(k, result, key_length)
            else:
                print(f"{k.ljust(key_length)}\t{result:.8f}")
    return results

def cov(v):
    return np.std(v, ddof=1) / np.average(v)

def get_key_length(data):
    return max(len(x) for x in data.keys())

def print_pvalue(k, result, key_length):
    print(f"{k.ljust(key_length)}\t{result:.4f}", end="")
    if result < TESTS_ALPHA:
        print(RED + " Rejected")
    else:
        print(GREEN + " OK")
    print(RESET, end="")
    
def print_table(data) -> None:
    def print_hr(lengths) -> None:
        print("-" * (sum(lengths) + 3*len(lengths) + 1))
    lengths = [0 for x in data[0]]
    for row in data:
        i = 0
        for cell in row:
            lengths[i] = max(lengths[i], len(str(cell)))
            i += 1
    i = 0
    print_hr(lengths)
    for row in data:
        for cell, l in zip(row, lengths):
            print("| " + str(cell).rjust(l), end=" ")
        print("|")
        if i == 0:
            print_hr(lengths)
        i += 1
    print_hr(lengths)


def print_tukey(data, k, result):
    key_length = get_key_length(data)
    shape = result.pvalue.shape
    out_matrix = [ [""] * (shape[0] + 1) for i in range(shape[0] + 1)]
    for i in range(shape[0]):
        cell = f"T{i+1}"
        out_matrix[0][i+1] = cell
        out_matrix[i+1][0] = cell
    out_matrix[0][0] = k.ljust(key_length)
    for i in range(shape[0]):
        for j in range(shape[0]):
            if i == j:
                out_matrix[i+1][j+1] = "---"
            else:
                out_matrix[i+1][j+1] = f"{result.pvalue[i][j]:.4f}"
    print_table(out_matrix)
    
    
def print_tukey_simple(data, k, result):
    shape = result.pvalue.shape
    out_matrix = [ [""] * (shape[0] + 1) for i in range(shape[0] + 1)]
    for i in range(shape[0]):
        cell = f"T{i+1}"
        out_matrix[0][i+1] = cell
        out_matrix[i+1][0] = cell
    out_matrix[0][0] = k
    for i in range(shape[0]):
        for j in range(shape[0]):
            if i == j:
                out_matrix[i+1][j+1] = "---"
            else:
                out_matrix[i+1][j+1] = f"{result.pvalue[i][j]:.4f}"
    for row in out_matrix:
        print("\t".join(row).replace(".", ","))

def do_tukey(data):
    results = []
    for k, v in data.items():
        result = stats.tukey_hsd(*v)
        results.append(result.pvalue)
        if PRINT_RESULTS:
            print_tukey(data, k, result)
    return results


def calculate_squares(data):
    results = {}
    for name, v in data.items():
        v_flat = flatten_list(v)
        
        N = len(v_flat)
        n = set([len(x) for x in v])
        if len(n) > 1:
            raise ValueError("Group sizes must be identical!")
        n = list(n)[0]
        k = len(v)
        
        avg_total = np.average(v_flat)
        avg_groups = [np.average(x) for x in v]
        
        ss_eff = n * sum(
            [(avg - avg_total) ** 2 for avg in avg_groups]
        )
        ss_res = sum(
            sum(
                (sample - avg) ** 2 for sample in group
            )
            for group, avg in zip(v, avg_groups)
        )
        ss_tot = ss_eff + ss_res
        
        df_eff = k - 1
        df_res = N - k
        
        ms_eff = ss_eff / (k - 1)
        ms_res = ss_res / df_res
        results[name] = {
            "ss_eff": ss_eff,
            "ss_res": ss_res,
            "ss_tot": ss_tot,
            "df_eff": df_eff,
            "df_res": df_res,
            "ms_eff": ms_eff,
            "ms_res": ms_res,
        }
    return results


def calculate_one_eta(values):
    flat = flatten_list(values)
    avg_total = np.average(flat)
    avg_groups = [np.average(x) for x in values]
    n = len(values[0])
    
    ss_eff = n * sum(
        [(avg - avg_total) ** 2 for avg in avg_groups]
    )
    ss_res = sum(
        sum(
            (sample - avg) ** 2 for sample in group
        )
        for group, avg in zip(values, avg_groups)
    )
    ss_tot = ss_eff + ss_res
    eta = ss_eff / ss_tot
    return eta


def calculate_one_omega(values):
    flat = flatten_list(values)
    avg_total = np.average(flat)
    avg_groups = [np.average(x) for x in values]
    n = len(values[0])
    
    ss_eff = n * sum(
        [(avg - avg_total) ** 2 for avg in avg_groups]
    )
    ss_res = sum(
        sum(
            (sample - avg) ** 2 for sample in group
        )
        for group, avg in zip(values, avg_groups)
    )
    ss_tot = ss_eff + ss_res
    
    df_eff = len(values)
    df_res = len(flat) - len(values)
    
    ms_res = ss_res / df_res
    
    top = ss_eff - df_eff * ms_res
    bot = ss_tot + ms_res
    omega = top / bot
    return omega


def calculate_eta(data):
    key_length = get_key_length(data)
    results = []
    for k, v in data.items():
        result = v["ss_eff"] / v["ss_tot"]
        results.append(result)
        if PRINT_RESULTS:
            print(f"{k.ljust(key_length)}\t{result:.4f}")
    return results

def calculate_omega(data):
    key_length = get_key_length(data)
    results = []
    for k, v in data.items():
        citatel = v["ss_eff"] - v["df_eff"] * v["ms_res"]
        jmenovatel = v["ss_tot"] + v["ms_res"]
        result = citatel / jmenovatel
        results.append(result)
        if PRINT_RESULTS:
            print(f"{k.ljust(key_length)}\t{result:.4f}")
    return results

def calculate_bootstrap_eta(data):
    key_length = get_key_length(data)
    results = []
    for k, v in data.items():
        np.random.seed(12345)
        etas = []
        for i in range(3000):
            vals = v[np.random.randint(v.shape[0], size=5), :]
            etas.append(calculate_one_eta(vals))
        interval = np.percentile(etas, [2.5, 97.5])
        results.append(interval)
        if PRINT_RESULTS:
            print(f"{k.ljust(key_length)}\t{interval[0]:.4f} - {interval[1]:.4f}")
    return results

def calculate_bootstrap_omega(data):
    key_length = get_key_length(data)
    results = []
    for k, v in data.items():
        np.random.seed(12345)
        etas = []
        for i in range(3000):
            vals = v[np.random.randint(v.shape[0], size=5), :]
            etas.append(calculate_one_omega(vals))
        interval = np.percentile(etas, [2.5, 97.5])
        results.append(interval)
        if PRINT_RESULTS:
            print(f"{k.ljust(key_length)}\t{interval[0]:.4f} - {interval[1]:.4f}")
    return results
        


#%%

def do_all_for_metric(metric, simple_names=True, unit_test=False):
    if unit_test:
        data = {"rnd_a": np.asarray([
            [0.744121715,0.71875,0.755244755,0.737609329],
            [0.793478261,0.823747681,0.777777778,0.816696915],
            [0.723004695,0.709677419,0.726717557,0.741433022],
            [0.80245023,0.811463047,0.788161994,0.768532526],
            [0.762857143,0.771186441,0.758915835,0.755304102]
        ])}
    else:
        data = load_all_methods(METHOD_NAMES, metric)
    
    df = pd.DataFrame({"method" if simple_names else "Split\nmethod": data.keys()})
    
    if PRINT_RESULTS:
            print("\nStep 1: Normality test / Shapiro-Wilk")
    res = do_tests(stats.shapiro, data, flatten=True, attr="pvalue")
    df['shapiro-wilk' if simple_names else 'Shapiro-Wilk\np-val'] = res
    
    if PRINT_RESULTS:
            print("\nStep 2: Homogeneity of variances test / Levene")
    res = do_tests(stats.levene, data, unpack=True, kw={"center": "mean"}, attr="pvalue")
    df['levene' if simple_names else 'Levene\np-val'] = res
    
    if PRINT_RESULTS:
            print("\nStep 5: Fold-fold homogeneity of means / Tukey HSD")
    tukey = do_tukey(data)
    n_bad_pairs = [np.sum(np.where(x < TESTS_ALPHA, True, False)) / 2
                   for x in tukey]
    df['tukey_pairs' if simple_names else 'Tukey\np<α pairs'] = n_bad_pairs
    if PRINT_RESULTS:
            print("Bootstrapped Omega-square 95% CI")
    res = calculate_bootstrap_omega(data)
    df['omega_sq_low' if simple_names else 'BS\'d ω²\n(2.5th\nPk)'] = [x[0] for x in res]
    # df['omega_sq_hi' if simple_names else 'ω²\n(high CI)'] = [x[1] for x in res]
    
    squares = calculate_squares(data)
    if PRINT_RESULTS:
            print("Omega-square")
    res = calculate_omega(squares)
    df['omega_squared' if simple_names else 'Naive\nω²'] = res
    
    if PRINT_RESULTS:
            print("\nStep 6: Total coefficient of variability")
    res = do_tests(cov, data, flatten=True)
    df['var_coeff' if simple_names else 'Variation\nCoeff.'] = res
    
    
    df["omega_rank" if simple_names else "ω²\nranking"] = df['omega_squared' if simple_names else "Naive\nω²"].rank(ascending=True)
    df["var_rank" if simple_names else "Var.coef.\nranking"] = df['var_coeff' if simple_names else 'Variation\nCoeff.'].rank(ascending=True)
    
    if PRINT_RESULTS:
            print("\nAverage metric value")
    res = do_tests(np.average, data, flatten=True)
    df['average' if simple_names else 'Average\nmetric val.'] = res
    
    if PRINT_RESULTS:
            print("\nStandard deviation")
    res = do_tests(np.std, data, flatten=True, kw={"ddof": 1})
    df['stdev' if simple_names else 'Standard\ndeviation'] = res
    
    results = {
        "tests": df,
        "tukey": tukey,
    }
    return metric, results


def rect_border(ws, x1, x2, y1, y2, inner_style, outer_style):
    for x in range(x1, x2+1):
        for y in range(y1, y2+1):
            border = xl.styles.Border(
                left=xl.styles.Side(border_style=outer_style if x == x1 else inner_style),
                right=xl.styles.Side(border_style=outer_style if x == x2 else inner_style),
                top=xl.styles.Side(border_style=outer_style if y == y1 else inner_style),
                bottom=xl.styles.Side(border_style=outer_style if y == y2 else inner_style),
            )
            cell = ws.cell(row=y, column=x)
            cell.border = border


def calc_and_save(metrics, out_name):
    wb = xl.Workbook()
    wb.remove(wb.active)
    
    tasks = Parallel(n_jobs=-1, return_as="generator")(
        delayed(do_all_for_metric)(metric)
        for metric in metrics
    )
    outputs = []
    for output in tasks:
        outputs.append(output)
        print("|", end="", flush=True)
    print("")
    order_map = {item: i for i, item in enumerate(metrics)}
    sorted_outputs = sorted(outputs, key=lambda x: order_map[x[0]])
    for metric, results in sorted_outputs:
        add_sheet(wb, metric, results["tests"], results["tukey"])
    wb.save(out_name)
    return {k: v for k, v in sorted_outputs}


def add_sheet(workbook, title, df, tukey, simple_names=True):
    ws = workbook.create_sheet(title)
    ws.title = title
    ws.sheet_view.zoomScale = 130
    ws.column_dimensions['A'].width = 13.4
    
    data = []
    data.append(list(df.columns))
    for _, row in df.iterrows():
        data.append(list(row))
        
    for row in data:
        ws.append(row)
    
    rows = len(data)
    for r in range(rows):
        cols = len(data[r])
        r += 1
        for c in range(cols):
            c += 1
            cell = ws.cell(row=r, column=c)
            if c == 1 and r > 1:
                cell.font = xl.styles.Font(italic=True)
            if r == 1:
                cell.font = xl.styles.Font(bold=True)
                cell.alignment = xl.styles.Alignment(wrapText=True)
            if c > 1 and r > 1:
                # p-values and effect sizes
                cell.number_format = "0.0000"
            if c in [4, 8, 9] and r > 1:
                # Tukey pairs
                cell.number_format = "0"
            if c == 7 and r > 1:
                # variation coefficient
                cell.number_format = "0.00%"
    rect_border(ws, 1, len(data[0]), 1, len(data), 'thin', 'medium')
    rect_border(ws, 1, len(data[0]), 1, 1, 'thin', 'medium')
    rect_border(ws, 1, 1, 1, len(data), 'thin', 'medium')
    rect_border(ws, 1, 1, 1, 1, 'thin', 'medium')
    
    
    green_fill = xl.styles.PatternFill(patternType="solid", bgColor="AAFFAA")
    yellow_fill = xl.styles.PatternFill(patternType="solid", bgColor="FFFFAA")
    orange_fill = xl.styles.PatternFill(patternType="solid", bgColor="FFD4AA")
    red_fill = xl.styles.PatternFill(patternType="solid", bgColor="FFAAAA")
    # p-values
    ws.conditional_formatting.add(f'B2:C{rows}', xl.formatting.rule.CellIsRule(
        operator="lessThan", formula=[f'{TESTS_ALPHA}'], fill=red_fill)
    )
    # Tukey
    ws.conditional_formatting.add(f'D2:D{rows}', xl.formatting.rule.CellIsRule(
        operator="greaterThan", formula=['0'], fill=red_fill)
    )
    # omega upper CI
    ws.conditional_formatting.add(f'E2:E{rows}', xl.formatting.rule.CellIsRule(
        operator="greaterThan", formula=[f'{OMEGA_LOWER_CI_LIMIT}'], fill=red_fill)
    )
    
    # omega-square: start at 0 (can have negative values but those are considered the same as 0)
    # var. coeff.: just use percentiles
    # st. dev.: just use percentiles
    for col, start in zip(['F', 'G', 'K'], ['num', 'percentile', 'percentile']):
        ws.conditional_formatting.add(
            f'{col}2:{col}{rows}',
            xl.formatting.rule.ColorScaleRule(
                start_type=start, start_value=0, start_color='AAFFAA',
                mid_type='percentile', mid_value=50, mid_color='FFFFAA',
                end_type='percentile', end_value=100, end_color='FFAAAA'
            )
        )
    
    # absolutely disgusting but I can't be bothered to go scour through openpyxl docs
    for col in ['H', 'I']:
        ws.conditional_formatting.add(f'{col}2:{col}{rows}', xl.formatting.rule.CellIsRule(
            operator="equal", formula=['1'], fill=green_fill)
        )
        ws.conditional_formatting.add(f'{col}2:{col}{rows}', xl.formatting.rule.CellIsRule(
            operator="equal", formula=['2'], fill=yellow_fill)
        )
        ws.conditional_formatting.add(f'{col}2:{col}{rows}', xl.formatting.rule.CellIsRule(
            operator="equal", formula=['3'], fill=orange_fill)
        )
        ws.conditional_formatting.add(f'{col}2:{col}{rows}', xl.formatting.rule.CellIsRule(
            operator="greaterThan", formula=['3'], fill=red_fill)
        )
    
    # average metric value - must check direction
    ws.conditional_formatting.add(
        f'J2:J{rows}',
        xl.formatting.rule.ColorScaleRule(
            start_type='percentile', start_value=0, start_color=('AAFFAA' if title in MINIMIZE_METRICS else 'FFAAAA'),
            mid_type='percentile', mid_value=50, mid_color='FFFFAA',
            end_type='percentile', end_value=100, end_color=('FFAAAA' if title in MINIMIZE_METRICS else 'AAFFAA')
        )
    )
        
        
    last_row = len(data)+2
    cell = ws.cell(row=last_row, column=1)
    cell.value = "Tukey pair tests"
    cell.font = xl.styles.Font(bold=True)
    last_row += 1
    
    for t, m in zip(tukey, data[1:]):
        app = []
        header = [m[0]]
        header.extend([f"T{x}" for x in range(1, 6)])
        app.append(header)
        for i, tr in enumerate(t):
            line = [f"T{i+1}"]
            line.extend(tr)
            line[i+1] = "---"
            app.append(line)
        for line in app:
            ws.append(line)
        rect_border(ws, 1, len(line), last_row, last_row+len(app)-1, 'thin', 'medium')
        ws.cell(row=last_row, column=1).font = xl.styles.Font(bold=True)
        for x in range(2, len(line)+1):
                ws.cell(row=last_row, column=x).font = xl.styles.Font(italic=True)
        for y in range(2, len(app)):
                ws.cell(row=last_row+y, column=1).font = xl.styles.Font(italic=True)
        ws.conditional_formatting.add(f'B{last_row}:F{last_row+len(app)-1}', xl.formatting.rule.CellIsRule(
            operator="lessThan", formula=[f'{TESTS_ALPHA}'], fill=red_fill)
        )
        for x in range(2, len(line)+1):
            for y in range(1, len(app)):
                ws.cell(row=last_row+y, column=x).number_format = "0.0000"
    
        last_row += len(app)
    
    ws["M1"] = "test_alpha" if simple_names else "Test α"
    ws["M2"] = TESTS_ALPHA
    ws["N1"] = "max_tukey_pairs" if simple_names else "Max. Tukey\npairs p<α"
    ws["N2"] = 0
    ws["O1"] = "omega_sq_low_thresh" if simple_names else "Max. ω² lower CI"
    ws["O2"] = OMEGA_LOWER_CI_LIMIT
    rect_border(ws, 13, 15, 1, 2, 'thin', 'medium')
    ws["M1"].alignment = xl.styles.Alignment(wrapText=True)
    ws["N1"].alignment = xl.styles.Alignment(wrapText=True)
    ws["O1"].alignment = xl.styles.Alignment(wrapText=True)

# unit_test = do_all_for_metric("", unit_test=True)
# tests = unit_test["tests"]

METRICS = ["ap_0.3_box", "ap_0.5_box", "ap_0.3", "ap_0.5", "ap_sum_0.3", "ap_sum_0.5", "fscore_avg_0.3", "fscore_avg_0.5", "mae", "rsq", "amae", "arsq"]
out_folder = "excels_2026-09-21_parallel"

# ROOT = Path("./metrics_2026-09-20_grouped_voxelrcnn")
# out = f"./{out_folder}/VoxelRCNN.xlsx"
# calc_and_save(METRICS, out)

# ROOT = Path("./metrics_2026-07-01_grouped_second")
# out = f"./{out_folder}/SECOND.xlsx"
# calc_and_save(METRICS, out)

# ROOT = Path("./metrics_2026-07-10_grouped_pointrcnn")
# out = f"./{out_folder}/PointRCNN.xlsx"
# calc_and_save(METRICS, out)

ROOT = Path("./metrics_2026-09-20_grouped_voxelrcnn")
out = f"./{out_folder}/voxelrcnn_p001.xlsx"
calc_and_save(METRICS, out)

ROOT = Path("./metrics_2026-07-01_grouped_second")
out = f"./{out_folder}/second_p001.xlsx"
calc_and_save(METRICS, out)

ROOT = Path("./metrics_2026-07-10_grouped_pointrcnn")
out = f"./{out_folder}/pointrcnn_p001.xlsx"
calc_and_save(METRICS, out)

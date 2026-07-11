import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import plotly.graph_objects as go
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy.interpolate import CubicSpline
from datetime import datetime
import os

st.set_page_config(page_title='Compressor Curve Regression', layout='wide')
st.title('Compressor Curve Regression Tool')

# ---------------------------------------------------------------------------
# Sidebar UI Configuration Controls
# ---------------------------------------------------------------------------
st.sidebar.header("Regression Settings")
method = st.sidebar.selectbox('Regression Method',
    ['Auto Best Fit','Linear','Quadratic','Cubic','4th Order','5th Order','Spline'])
points = st.sidebar.slider('Generated Points', 15, 50, 15)

st.sidebar.markdown("---")
st.sidebar.header("Target Units Configuration")

# Dropdown overrides for Curve and Operating Parameters
target_flow = st.sidebar.selectbox('Target Volumetric Flow Unit', ['m3/hr', 'cfm', 'm3/min'], index=0)
target_head = st.sidebar.selectbox('Target Head Unit', ['m', 'ft', 'kj/kg'], index=0)
target_power = st.sidebar.selectbox('Target Power Unit', ['kW', 'hp', 'btu/hr'], index=0)
target_eff = st.sidebar.selectbox('Target Efficiency Unit', ['%', 'fraction'], index=0)
target_temp = st.sidebar.selectbox('Target Temperature Unit', ['deg C', 'deg F', 'K'], index=0)
target_press = st.sidebar.selectbox('Target Pressure Unit', ['kg/cm2a', 'bara', 'psia', 'barg', 'psig'], index=0)
target_diameter = st.sidebar.selectbox('Target Diameter Unit', ['m', 'mm', 'inch'], index=0)

file = st.file_uploader('Upload Workbook', type=['xlsx'])

# ---------------------------------------------------------------------------
# Physical constants and base unit conversions
# ---------------------------------------------------------------------------
R_UNIVERSAL = 8314.462618   # J/(kmol.K)
G = 9.80665                 # m/s^2

def normalize_unit(u):
    """Fold formatting variants ('m³/hr', 'M3/HR', 'ft^3 / min', 'Ft3.Min') onto one key."""
    s = str(u).strip().lower()
    s = s.replace('³', '3').replace('²', '2')
    s = s.replace('^3', '3').replace('^2', '2')
    s = s.replace(' ', '').replace('.', '').replace('-', '').replace('_', '')
    s = s.replace('cu', '')
    return s

# Conversion maps referenced to baseline absolute SI/Metric values
FLOW_TO_M3HR = {
    'cfm': 1.699010796, 'acfm': 1.699010796, 'icfm': 1.699010796,
    'ft3/min': 1.699010796, 'ft3min': 1.699010796, 'ft/min': 1.699010796, 'cf/min': 1.699010796,
    'cfh': 0.028316847, 'ft3/hr': 0.028316847, 'ft3/h': 0.028316847, 'ft3hr': 0.028316847,
    'ft/hr': 0.028316847, 'cf/hr': 0.028316847,
    'cfs': 101.9406, 'ft3/s': 101.9406, 'ft3s': 101.9406, 'ft/s': 101.9406, 'cf/s': 101.9406,
    'm3/hr': 1.0, 'm3/h': 1.0, 'm3hr': 1.0, 'm3h': 1.0,
    'm3/min': 60.0, 'm3min': 60.0, 'm3/s': 3600.0, 'm3s': 3600.0,
    'l/min': 0.06, 'lpm': 0.06, 'l/s': 3.6, 'lps': 3.6, 'l/hr': 0.001, 'lph': 0.001,
    'gpm': 0.227124707, 'usgpm': 0.227124707, 'galmin': 0.227124707,
    'igpm': 0.272765, 'ukgpm': 0.272765, 'impgpm': 0.272765,
    'gph': 0.003785412, 'usgph': 0.003785412,
    'bbl/day': 0.006624459, 'bpd': 0.006624459, 'bbl/d': 0.006624459,
    'mmscfd': 1179.874,
}

HEAD_TO_M = {
    'ft': 0.3048, 'feet': 0.3048, 'foot': 0.3048,
    'ftlbf/lbm': 0.3048, 'lbfft/lbm': 0.3048, 'ftlb/lb': 0.3048,
    'in': 0.0254, 'inch': 0.0254, 'inches': 0.0254,
    'm': 1.0, 'meter': 1.0, 'metre': 1.0, 'meters': 1.0, 'metres': 1.0,
    'mm': 0.001,
    'kj/kg': 101.9716, 'j/kg': 0.1019716,
    'btu/lb': 237.2075,
}

POWER_TO_KW = {
    'hp': 0.745699872, 'bhp': 0.745699872, 'mechhp': 0.745699872, 'hp(i)': 0.745699872,
    'ps': 0.735499, 'cv': 0.735499, 'metrichp': 0.735499, 'hp(m)': 0.735499,
    'kw': 1.0, 'w': 0.001, 'mw': 1000.0,
    'btu/hr': 0.000293071, 'btu/h': 0.000293071, 'btuh': 0.000293071,
    'btu/s': 1.055056, 'btus': 1.055056,
    'ftlb/s': 0.001355818, 'ftlbf/s': 0.001355818,
    'kcal/hr': 0.001163, 'kcal/h': 0.001163,
}

EFF_TO_PCT = {
    '%': 1.0, 'pct': 1.0, 'percent': 1.0, 'percentage': 1.0,
    'fraction': 100.0, 'decimal': 100.0, 'ratio': 100.0, 'frac': 100.0,
}

LENGTH_TO_M = {
    'm': 1.0, 'meter': 1.0, 'meters': 1.0, 'metre': 1.0,
    'mm': 0.001, 'millimeter': 0.001, 'millimeters': 0.001,
    'cm': 0.01, 'centimeter': 0.01, 'centimeters': 0.01,
    'in': 0.0254, 'inch': 0.0254, 'inches': 0.0254, '"': 0.0254,
    'ft': 0.3048, 'feet': 0.3048, 'foot': 0.3048, "'": 0.3048
}

# ---------------------------------------------------------------------------
# Dynamic UOM Conversion Core Routing Logic
# ---------------------------------------------------------------------------
def convert_to_si_base(val, incoming_unit, mapping_table):
    """Helper to convert an input value back to the SI/Metric reference line."""
    key = normalize_unit(incoming_unit)
    if key in mapping_table:
        return val * mapping_table[key], True
    return val, False

def convert_from_si_base(val, desired_target, mapping_table):
    """Helper to convert from an SI/Metric reference line to any selected output format."""
    key = normalize_unit(desired_target)
    if key in mapping_table:
        return val / mapping_table[key]
    return val

def route_temperature(val, from_unit, to_unit):
    """Transforms temperature calculations across any combination of inputs/outputs."""
    u_from = normalize_unit(from_unit)
    u_to = normalize_unit(to_unit)
    
    # Base normalization step: convert input to Celsius
    if u_from in ['c', 'degc', 'celsius']: c = val
    elif u_from in ['f', 'degf', 'fahrenheit']: c = (val - 32) * 5.0 / 9.0
    elif u_from in ['k', 'kelvin']: c = val - 273.15
    elif u_from in ['r', 'rankine']: c = (val - 491.67) * 5.0 / 9.0
    else: return val, False # Unrecognized incoming unit

    # Conversion step: convert Celsius to requested layout
    if u_to in ['c', 'degc', 'celsius']: return c, True
    if u_to in ['f', 'degf', 'fahrenheit']: return (c * 9.0 / 5.0) + 32, True
    if u_to in ['k', 'kelvin']: return c + 273.15, True
    if u_to in ['r', 'rankine']: return (c * 9.0 / 5.0) + 491.67, True
    return c, True

def route_pressure(val, from_unit, to_unit):
    """Transforms absolute and gauge pressure properties dynamically."""
    u_from = normalize_unit(from_unit)
    u_to = normalize_unit(to_unit)
    
    # Base step: convert any incoming variance to absolute bara
    if u_from in ['bar', 'bara']: bara = val
    elif u_from in ['barg']: bara = val + 1.01325
    elif u_from in ['psi', 'psia', 'lbf/in2']: bara = val * 0.0689476
    elif u_from in ['psig']: bara = (val + 14.6959) * 0.0689476
    elif u_from in ['kg/cm2', 'kg/cm2a', 'kgcm2', 'kgcm2a', 'ata']: bara = val * 0.980665
    elif u_from in ['kg/cm2g', 'kgcm2g', 'atg']: bara = (val + 1.03323) * 0.980665
    elif u_from in ['kpa', 'kpaa']: bara = val / 100.0
    elif u_from in ['kpag']: bara = (val + 101.325) / 100.0
    elif u_from in ['mpa', 'mpaa']: bara = val * 10.0
    elif u_from in ['mpag']: bara = (val * 10.0) + 1.01325
    elif u_from in ['atm']: bara = val * 1.01325
    elif u_from in ['torr', 'mmhg']: bara = val / 750.062
    else: return val, False

    # Conversion step: transform absolute bara to requested target unit
    if u_to in ['bar', 'bara']: return bara, True
    if u_to in ['barg']: return bara - 1.01325, True
    if u_to in ['psi', 'psia', 'lbf/in2']: return bara / 0.0689476, True
    if u_to in ['psig']: return (bara / 0.0689476) - 14.6959, True
    if u_to in ['kg/cm2', 'kg/cm2a', 'kgcm2', 'kgcm2a', 'ata']: return bara / 0.980665, True
    if u_to in ['kg/cm2g', 'kgcm2g', 'atg']: return (bara / 0.980665) - 1.03323, True
    if u_to in ['kpa', 'kpaa']: return bara * 100.0, True
    if u_to in ['kpag']: return (bara * 100.0) - 101.325, True
    if u_to in ['mpa', 'mpaa']: return bara / 10.0, True
    if u_to in ['mpag']: return (bara - 1.01325) / 10.0, True
    if u_to in ['atm']: return bara / 1.01325, True
    if u_to in ['torr', 'mmhg']: return bara * 750.062, True
    return bara, True

def gas_density_kg_m3(pressure_any, temp_any, p_unit, t_unit, mw, z):
    """Calculates thermodynamic gas density by safely converting any UI setup to standard metric base."""
    p_bara, _ = route_pressure(pressure_any, p_unit, 'bara')
    t_c, _ = route_temperature(temp_any, t_unit, 'deg C')
    
    p_pa = p_bara * 100000.0
    t_k = t_c + 273.15
    return (p_pa * mw) / (z * R_UNIVERSAL * t_k)

# ---------------------------------------------------------------------------
# Core UI Data Extraction Logic
# ---------------------------------------------------------------------------
def clean_parameter_name(name):
    n = str(name).lower()
    if 'head' in n: return 'Head'
    if 'eff' in n: return 'Efficiency'
    if 'power' in n or 'bhp' in n or 'kw' in n: return 'Power'
    return str(name)

def detect_triplet_blocks(raw_df):
    blocks = []
    rows, cols = raw_df.shape
    for r in range(rows):
        for c in range(cols - 2):
            v1 = str(raw_df.iloc[r, c]).strip().lower()
            v2 = str(raw_df.iloc[r, c + 1]).strip().lower()
            v3 = str(raw_df.iloc[r, c + 2]).strip()
            if v1 == 'speed' and 'flow' in v2:
                p = clean_parameter_name(v3)
                if p.lower() != 'nan':
                    speed_unit = flow_unit = value_unit = ''
                    if r + 1 < rows:
                        speed_unit = str(raw_df.iloc[r + 1, c]).strip()
                        flow_unit = str(raw_df.iloc[r + 1, c + 1]).strip()
                        value_unit = str(raw_df.iloc[r + 1, c + 2]).strip()
                    blocks.append({
                        'parameter': p, 'header_row': r, 'start_col': c,
                        'speed_unit': speed_unit, 'flow_unit': flow_unit,
                        'value_unit': value_unit
                    })
    uniq = []
    seen = set()
    for b in blocks:
        k = (b['parameter'], b['start_col'])
        if k not in seen:
            seen.add(k)
            uniq.append(b)
    return uniq

def extract_block_data(raw_df, block):
    """Extract and map curve block datasets straight to user-configured UI targets."""
    r = block['header_row']
    c = block['start_col']
    data = []
    for row in range(r + 1, len(raw_df)):
        try:
            sp = float(raw_df.iloc[row, c])
            fl = float(raw_df.iloc[row, c + 1])
            val = float(raw_df.iloc[row, c + 2])
            data.append([sp, fl, val])
        except:
            pass
    df = pd.DataFrame(data, columns=['Speed', 'Flow', 'Value'])
    if df.empty:
        return df, False, False

    # 1. Normalize Flow to standard m3/hr, then map to user chosen target layout
    flow_si, flow_converted = convert_to_si_base(df['Flow'], block['flow_unit'], FLOW_TO_M3HR)
    df['Flow'] = convert_from_si_base(flow_si, target_flow, FLOW_TO_M3HR)

    # 2. Normalize and route curve values based on context
    param = block['parameter']
    value_converted = False
    
    if param == 'Head':
        val_si, value_converted = convert_to_si_base(df['Value'], block['value_unit'], HEAD_TO_M)
        df['Value'] = convert_from_si_base(val_si, target_head, HEAD_TO_M)
    elif param == 'Power':
        val_si, value_converted = convert_to_si_base(df['Value'], block['value_unit'], POWER_TO_KW)
        df['Value'] = convert_from_si_base(val_si, target_power, POWER_TO_KW)
    elif param == 'Efficiency':
        val_si, value_converted = convert_to_si_base(df['Value'], block['value_unit'], EFF_TO_PCT)
        df['Value'] = convert_from_si_base(val_si, target_eff, EFF_TO_PCT)

    return df, flow_converted, value_converted

def detect_property_block(raw_df):
    rows, cols = raw_df.shape
    for r in range(rows):
        for c in range(cols - 2):
            v1 = str(raw_df.iloc[r, c]).strip().lower()
            v2 = str(raw_df.iloc[r, c + 1]).strip().lower()
            v3 = str(raw_df.iloc[r, c + 2]).strip().lower()
            if v1 == 'parameter' and v2 == 'value' and v3 == 'units':
                return {'header_row': r, 'start_col': c}
    return None

def extract_property_block(raw_df, block):
    if block is None:
        return pd.DataFrame(columns=['Parameter', 'Value', 'Units'])
    r = block['header_row']
    c = block['start_col']
    rows_out = []
    for row in range(r + 1, len(raw_df)):
        param = raw_df.iloc[row, c]
        value = raw_df.iloc[row, c + 1]
        units = raw_df.iloc[row, c + 2]
        if pd.isna(param) or str(param).strip() == '':
            break
            
        param_str = str(param).strip()
        unit_str = '' if pd.isna(units) else str(units).strip()
        val_float = value.item() if hasattr(value, 'item') else value
        
        try:
            val_float = float(val_float)
            p_lower = param_str.lower()
            
            # Dynamically convert properties based on UI selections
            if 'temperature' in p_lower:
                val_float, converted = route_temperature(val_float, unit_str, target_temp)
                if converted: unit_str = target_temp
            elif 'pressure' in p_lower:
                val_float, converted = route_pressure(val_float, unit_str, target_press)
                if converted: unit_str = target_press
            elif 'diameter' in p_lower:
                val_si, converted = convert_to_si_base(val_float, unit_str, LENGTH_TO_M)
                val_float = convert_from_si_base(val_si, target_diameter, LENGTH_TO_M)
                if converted: unit_str = target_diameter
        except (ValueError, TypeError):
            pass
            
        rows_out.append({
            'Parameter': param_str,
            'Value': val_float,
            'Units': unit_str
        })
    return pd.DataFrame(rows_out, columns=['Parameter', 'Value', 'Units'])

# ---------------------------------------------------------------------------
# Mathematical Regression Modeling Core Engines
# ---------------------------------------------------------------------------
def build_model(x, y, meth):
    if meth == 'Spline':
        idx = np.argsort(x)
        x = x[idx]; y = y[idx]
        s = CubicSpline(x, y)
        r2 = r2_score(y, s(x))
        return {'type': 'spline', 'model': s, 'xmin': x.min(), 'xmax': x.max(), 'r2': r2}

    deg = {'Linear': 1, 'Quadratic': 2, 'Cubic': 3, '4th Order': 4, '5th Order': 5}[meth]
    poly = PolynomialFeatures(deg)
    X = poly.fit_transform(x.reshape(-1, 1))
    lr = LinearRegression().fit(X, y)
    r2 = r2_score(y, lr.predict(X))
    return {'type': 'poly', 'poly': poly, 'model': lr, 'xmin': x.min(), 'xmax': x.max(), 'r2': r2}

def predict_model(obj, flow):
    if obj['type'] == 'spline':
        return obj['model'](flow)
    return obj['model'].predict(obj['poly'].transform(flow.reshape(-1, 1)))

def auto_best(x, y):
    best = None; best_name = None; best_r2 = -1e9
    for m in ['Linear', 'Quadratic', 'Cubic', '4th Order', '5th Order', 'Spline']:
        try:
            mdl = build_model(x, y, m)
            if mdl['r2'] > best_r2:
                best_r2 = mdl['r2']; best = mdl; best_name = m
        except:
            pass
    return best_name, best

def gas_properties_from_df(prop_df):
    """Safely extracts process fields dynamically from localized operating blocks."""
    lookup = {}
    for _, row in prop_df.iterrows():
        p_norm = row['Parameter'].strip().lower()
        if 'pressure' in p_norm:
            lookup['p_val'] = row['Value']
            lookup['p_unit'] = row['Units']
        elif 'temperature' in p_norm:
            lookup['t_val'] = row['Value']
            lookup['t_unit'] = row['Units']
        elif 'weight' in p_norm or 'mw' in p_norm:
            lookup['mw'] = row['Value']
        elif 'compressibility' in p_norm or 'z' in p_norm:
            lookup['z'] = row['Value']
            
    try:
        return {
            'p_val': float(lookup['p_val']),
            'p_unit': lookup['p_unit'],
            't_val': float(lookup['t_val']),
            't_unit': lookup['t_unit'],
            'mw': float(lookup['mw']),
            'z': float(lookup['z']),
        }
    except (KeyError, TypeError, ValueError):
        return None

def compute_missing_parameter(available, common_flow, gas_props):
    """Calculates missing curve parameters using raw metrics before mapping back to target units."""
    have = set(available.keys())
    needed = {'Head', 'Efficiency', 'Power'} - have
    if len(needed) != 1 or gas_props is None:
        return None, None
    missing = needed.pop()

    # 1. Convert operational inputs back to metric SI base for calculation safety
    flow_m3hr = convert_to_si_base(common_flow, target_flow, FLOW_TO_M3HR)[0]
    rho = gas_density_kg_m3(gas_props['p_val'], gas_props['t_val'], gas_props['p_unit'], gas_props['t_unit'], gas_props['mw'], gas_props['z'])
    mass_flow_kg_s = flow_m3hr * rho / 3600.0

    # Convert available curves parameters back to basic SI units
    head_m = convert_to_si_base(available.get('Head', 0.0), target_head, HEAD_TO_M)[0]
    power_kw = convert_to_si_base(available.get('Power', 0.0), target_power, POWER_TO_KW)[0]
    eff_pct = convert_to_si_base(available.get('Efficiency', 0.0), target_eff, EFF_TO_PCT)[0]

    # 2. Complete thermodynamic checks
    if missing == 'Power':
        eff_frac = eff_pct / 100.0
        calc_kw = mass_flow_kg_s * G * head_m / eff_frac / 1000.0
        return 'Power', convert_from_si_base(calc_kw, target_power, POWER_TO_KW)

    if missing == 'Head':
        eff_frac = eff_pct / 100.0
        calc_m = power_kw * 1000.0 * eff_frac / (mass_flow_kg_s * G)
        return 'Head', convert_from_si_base(calc_m, target_head, HEAD_TO_M)

    if missing == 'Efficiency':
        calc_pct = (mass_flow_kg_s * G * head_m) / (power_kw * 1000.0) * 100.0
        return 'Efficiency', convert_from_si_base(calc_pct, target_eff, EFF_TO_PCT)

    return None, None

# ---------------------------------------------------------------------------
# Streamlit Execution Pipeline Engine
# ---------------------------------------------------------------------------
if file:
    xls = pd.ExcelFile(file)
    output = BytesIO()
    r2_rows = []
    overview = []
    property_rows = []

    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        for stage in xls.sheet_names:
            st.header(stage)
            raw = pd.read_excel(xls, sheet_name=stage, header=None)
            blocks = detect_triplet_blocks(raw)

            prop_block = detect_property_block(raw)
            prop_df = extract_property_block(raw, prop_block)
            gas_props = None
            if not prop_df.empty:
                st.subheader(f'Operating Conditions (Standardized to {target_temp}, {target_press})')
                st.dataframe(prop_df, use_container_width=True)
                for _, row in prop_df.iterrows():
                    property_rows.append([stage, row['Parameter'], row['Value'], row['Units']])
                gas_props = gas_properties_from_df(prop_df)
            else:
                st.warning(f'No operating-conditions block found in {stage}')

            if not blocks:
                st.warning(f'No blocks found in {stage}')
                continue

            block_summary = pd.DataFrame([
                {'parameter': b['parameter'], 'speed_unit': b['speed_unit'],
                 'flow_unit': b['flow_unit'], 'value_unit': b['value_unit']}
                for b in blocks
            ])
            st.dataframe(block_summary)

            stage_models = {}
            stage_parameters = []
            tabs = st.tabs([b['parameter'] for b in blocks])

            for tab, block in zip(tabs, blocks):
                with tab:
                    param = block['parameter']
                    df, flow_conv, val_conv = extract_block_data(raw, block)
                    stage_parameters.append(param)

                    unit_label = {'Head': target_head, 'Power': target_power, 'Efficiency': target_eff}.get(param, '')
                    st.caption(f"Visualizing with user choices -> Flow: {target_flow} | {param}: {unit_label}")

                    fig = go.Figure()
                    if param not in stage_models:
                        stage_models[param] = {}

                    for speed in sorted(df['Speed'].unique()):
                        sdf = df[df['Speed'] == speed]
                        if len(sdf) < 4:
                            continue

                        x = sdf['Flow'].values.astype(float)
                        y = sdf['Value'].values.astype(float)

                        if method == 'Auto Best Fit':
                            used, mdl = auto_best(x, y)
                        else:
                            mdl = build_model(x, y, method)
                            used = method

                        stage_models[param][speed] = mdl
                        r2_rows.append([stage, speed, param, used, round(mdl['r2'], 6)])

                        flow_fit = np.linspace(x.min(), x.max(), points)
                        y_fit = predict_model(mdl, flow_fit)

                        fig.add_trace(go.Scatter(x=x, y=y, mode='markers', name=f'{speed} Original'))
                        fig.add_trace(go.Scatter(x=flow_fit, y=y_fit, mode='lines', name=f'{speed} Fit'))

                    st.plotly_chart(fig, use_container_width=True)

            speeds = set()
            for p in stage_models:
                speeds.update(stage_models[p].keys())

            export_rows = []
            computed_param_name = None

            for speed in sorted(speeds):
                available = []
                for p in stage_models:
                    if speed in stage_models[p]:
                        available.append(stage_models[p][speed])

                if len(available) < 1:
                    continue

                common_min = max(m['xmin'] for m in available)
                common_max = min(m['xmax'] for m in available)

                if common_max <= common_min:
                    continue

                common_flow = np.linspace(common_min, common_max, points)
                temp = {'Speed': [speed] * points, f'Flow ({target_flow})': common_flow}

                predicted_at_speed = {}
                for p in stage_models:
                    if speed in stage_models[p]:
                        vals = predict_model(stage_models[p][speed], common_flow)
                        predicted_at_speed[p] = vals
                        lbl = {'Head': target_head, 'Power': target_power, 'Efficiency': target_eff}.get(p, '')
                        temp[f'{p} ({lbl})'] = vals

                if gas_props is not None:
                    name, values = compute_missing_parameter(predicted_at_speed, common_flow, gas_props)
                    if name is not None:
                        computed_param_name = name
                        lbl = {'Head': target_head, 'Power': target_power, 'Efficiency': target_eff}.get(name, '')
                        temp[f'{name} ({lbl}, calculated)'] = values

                export_rows.append(pd.DataFrame(temp))

            if computed_param_name:
                st.success(f"Calculated missing curve parameter **{computed_param_name}** for {stage}.")

            if export_rows:
                final_df = pd.concat(export_rows, ignore_index=True)
                final_df.to_excel(writer, sheet_name=stage[:31], index=False)

            overview.append({
                'Stage': stage,
                'Parameters': ','.join(stage_parameters),
                'Blocks Found': len(blocks),
                'Calculated Parameter': computed_param_name or ''
            })

        pd.DataFrame(r2_rows, columns=['Stage', 'Speed', 'Parameter', 'Method', 'R2']).to_excel(writer, sheet_name='Summary_R2', index=False)
        pd.DataFrame(overview).to_excel(writer, sheet_name='Workbook_Overview', index=False)

        if property_rows:
            pd.DataFrame(property_rows, columns=['Stage', 'Parameter', 'Value', 'Units']).to_excel(writer, sheet_name='Operating_Conditions', index=False)

    output.seek(0)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    input_filename = os.path.splitext(file.name)[0]
    output_filename = f"{input_filename}_Custom_UOM_Output_{timestamp}.xlsx"

    st.download_button(
        label="Download Customized Workbook",
        data=output.getvalue(),
        file_name=output_filename,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

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
st.sidebar.header("UOM Configuration (Target / Fallback Overrides)")

# Dropdown selections act as fallback if file auto-detect fails, and dictate final export format
target_flow = st.sidebar.selectbox('Volumetric Flow Unit', ['m3/hr', 'cfm', 'm3/min'], index=0)
target_head = st.sidebar.selectbox('Head Unit', ['m', 'ft', 'kj/kg'], index=0)
target_power = st.sidebar.selectbox('Power Unit', ['kW', 'hp', 'btu/hr'], index=0)
target_eff = st.sidebar.selectbox('Efficiency Unit', ['%', 'fraction'], index=0)
target_temp = st.sidebar.selectbox('Temperature Unit', ['deg C', 'deg F', 'K'], index=0)
target_press = st.sidebar.selectbox('Pressure Unit', ['kg/cm2a', 'bara', 'psia', 'barg', 'psig'], index=0)
target_diameter = st.sidebar.selectbox('Diameter Unit', ['m', 'mm', 'inch'], index=0)

file = st.file_uploader('Upload Workbook', type=['xlsx'])

# ---------------------------------------------------------------------------
# Physical constants and base unit conversions
# ---------------------------------------------------------------------------
R_UNIVERSAL = 8314.462618 # J/(kmol.K)
G = 9.80665                 # m/s^2

def normalize_unit(u):
    """Fold formatting variants ('m³/hr', 'M3/HR', 'ft^3 / min', 'Ft3.Min') onto one key."""
    s = str(u).strip().lower()
    s = s.replace('³', '3').replace('²', '2')
    s = s.replace('^3', '3').replace('^2', '2')
    s = s.replace(' ', '').replace('.', '').replace('-', '').replace('_', '')
    s = s.replace('cu', '')
    return s

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
    'ft': 0.3048, 'feet': 0.3048, 'foot': 0.3048, 'ftlbf/lbm': 0.3048, 'lbfft/lbm': 0.3048, 'ftlb/lb': 0.3048,
    'in': 0.0254, 'inch': 0.0254, 'inches': 0.0254, 'm': 1.0, 'meter': 1.0, 'metre': 1.0, 'meters': 1.0, 'metres': 1.0, 'mm': 0.001,
    'kj/kg': 101.9716, 'j/kg': 0.1019716, 'btu/lb': 237.2075,
}

POWER_TO_KW = {
    'hp': 0.745699872, 'bhp': 0.745699872, 'mechhp': 0.745699872, 'hp(i)': 0.745699872,
    'ps': 0.735499, 'cv': 0.735499, 'metrichp': 0.735499, 'hp(m)': 0.735499, 'kw': 1.0, 'w': 0.001, 'mw': 1000.0,
    'btu/hr': 0.000293071, 'btu/h': 0.000293071, 'btuh': 0.000293071, 'btu/s': 1.055056, 'btus': 1.055056,
    'ftlb/s': 0.001355818, 'ftlbf/s': 0.001355818, 'kcal/hr': 0.001163, 'kcal/h': 0.001163,
}

EFF_TO_PCT = {
    '%': 1.0, 'pct': 1.0, 'percent': 1.0, 'percentage': 1.0, 'fraction': 100.0, 'decimal': 100.0, 'ratio': 100.0, 'frac': 100.0,
}

LENGTH_TO_M = {
    'm': 1.0, 'meter': 1.0, 'meters': 1.0, 'metre': 1.0, 'mm': 0.001, 'millimeter': 0.001, 'millimeters': 0.001,
    'cm': 0.01, 'centimeter': 0.01, 'centimeters': 0.01, 'in': 0.0254, 'inch': 0.0254, 'inches': 0.0254, '"': 0.0254,
    'ft': 0.3048, 'feet': 0.3048, 'foot': 0.3048, "'": 0.3048
}

# ---------------------------------------------------------------------------
# Base Dynamic Mapping Routing Logic (Internal Base = SI Metrics)
# ---------------------------------------------------------------------------
def convert_to_si_base(val, incoming_unit, mapping_table, fallback_unit=None):
    """Convert value to internal SI baseline. Uses fallback if auto-detection fails."""
    key = normalize_unit(incoming_unit)
    if key in mapping_table:
        return val * mapping_table[key], True, incoming_unit
    if fallback_unit:
        fb_key = normalize_unit(fallback_unit)
        if fb_key in mapping_table:
            return val * mapping_table[fb_key], False, fallback_unit
    return val, False, incoming_unit

def convert_from_si_base(val, desired_target, mapping_table):
    """Convert value from internal SI baseline back to chosen target UOM."""
    key = normalize_unit(desired_target)
    if key in mapping_table:
        return val / mapping_table[key]
    return val

def route_temperature_to_c(val, from_unit, fallback_unit=None):
    """Always resolves temperature down to internal standard Celsius."""
    u_from = normalize_unit(from_unit)
    is_detected = u_from in ['c', 'degc', 'celsius', 'f', 'degf', 'fahrenheit', 'k', 'kelvin', 'r', 'rankine']
    
    if not is_detected and fallback_unit:
        u_from = normalize_unit(fallback_unit)
        from_unit = fallback_unit

    if u_from in ['c', 'degc', 'celsius']: return val, is_detected, from_unit
    if u_from in ['f', 'degf', 'fahrenheit']: return (val - 32) * 5.0 / 9.0, is_detected, from_unit
    if u_from in ['k', 'kelvin']: return val - 273.15, is_detected, from_unit
    if u_from in ['r', 'rankine']: return (val - 491.67) * 5.0 / 9.0, is_detected, from_unit
    return val, False, from_unit

def route_pressure_to_bara(val, from_unit, fallback_unit=None):
    """Always resolves pressure metrics down to absolute bara."""
    u_from = normalize_unit(from_unit)
    
    # Validations list
    valid_units = ['bar', 'bara', 'barg', 'psi', 'psia', 'lbf/in2', 'psig', 'kg/cm2', 'kg/cm2a', 
                   'kgcm2', 'kgcm2a', 'ata', 'kg/cm2g', 'kgcm2g', 'atg', 'kpa', 'kpaa', 'kpag', 'mpa', 'mpaa', 'mpag', 'atm', 'torr', 'mmhg']
    is_detected = u_from in valid_units
    
    if not is_detected and fallback_unit:
        u_from = normalize_unit(fallback_unit)
        from_unit = fallback_unit

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
    else: return val, False, from_unit

    return bara, is_detected, from_unit

def route_pressure_from_bara(bara, to_unit):
    """Converts back from absolute bara baseline to requested format configuration."""
    u_to = normalize_unit(to_unit)
    if u_to in ['bar', 'bara']: return bara
    if u_to in ['barg']: return bara - 1.01325
    if u_to in ['psi', 'psia', 'lbf/in2']: return bara / 0.0689476
    if u_to in ['psig']: return (bara / 0.0689476) - 14.6959
    if u_to in ['kg/cm2', 'kg/cm2a', 'kgcm2', 'kgcm2a', 'ata']: return bara / 0.980665
    if u_to in ['kg/cm2g', 'kgcm2g', 'atg']: return (bara / 0.980665) - 1.03323
    if u_to in ['kpa', 'kpaa']: return bara * 100.0
    if u_to in ['kpag']: return (bara * 100.0) - 101.325
    if u_to in ['mpa', 'mpaa']: return bara / 10.0
    if u_to in ['mpag']: return (bara - 1.01325) / 10.0
    if u_to in ['atm']: return bara / 1.01325
    if u_to in ['torr', 'mmhg']: return bara * 750.062
    return bara

def route_temperature_from_c(c, to_unit):
    """Converts back from Celsius baseline to requested layout choice."""
    u_to = normalize_unit(to_unit)
    if u_to in ['c', 'degc', 'celsius']: return c
    if u_to in ['f', 'degf', 'fahrenheit']: return (c * 9.0 / 5.0) + 32
    if u_to in ['k', 'kelvin']: return c + 273.15
    if u_to in ['r', 'rankine']: return (c * 9.0 / 5.0) + 491.67
    return c

def gas_density_kg_m3(p_bara, t_c, mw, z):
    """Calculates thermodynamic gas density safely using SI internal metrics."""
    p_pa = p_bara * 100000.0
    t_k = t_c + 273.15
    return (p_pa * mw) / (z * R_UNIVERSAL * t_k)

# ---------------------------------------------------------------------------
# Data Extraction & Regression Pipelines
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
    """Extract and map curve block datasets straight to absolute internal SI metrics."""
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
        return df, False, False, block['flow_unit'], block['value_unit']

    # Normalize values into standard SI variables for safe curve-fitting calculations
    flow_m3hr, flow_conv, used_flow_u = convert_to_si_base(df['Flow'], block['flow_unit'], FLOW_TO_M3HR, fallback_unit=target_flow)
    df['Flow'] = flow_m3hr

    param = block['parameter']
    val_conv = False
    used_val_u = block['value_unit']
    
    if param == 'Head':
        df['Value'], val_conv, used_val_u = convert_to_si_base(df['Value'], block['value_unit'], HEAD_TO_M, fallback_unit=target_head)
    elif param == 'Power':
        df['Value'], val_conv, used_val_u = convert_to_si_base(df['Value'], block['value_unit'], POWER_TO_KW, fallback_unit=target_power)
    elif param == 'Efficiency':
        df['Value'], val_conv, used_val_u = convert_to_si_base(df['Value'], block['value_unit'], EFF_TO_PCT, fallback_unit=target_eff)

    return df, flow_conv, val_conv, used_flow_u, used_val_u

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
    """Processes Operating Conditions sheet block and matches internal reference units cleanly."""
    if block is None:
        return pd.DataFrame(columns=['Parameter', 'Value', 'Units', 'Internal_SI_Val', 'Internal_SI_Unit'])
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
        
        si_val = val_float
        si_unit = unit_str
        
        try:
            val_float = float(val_float)
            p_lower = param_str.lower()
            
            if 'temperature' in p_lower:
                si_val, converted, resolved_u = route_temperature_to_c(val_float, unit_str, fallback_unit=target_temp)
                si_unit = 'deg C'
                unit_str = resolved_u if converted else f"{resolved_u} (Override)"
            elif 'pressure' in p_lower:
                si_val, converted, resolved_u = route_pressure_to_bara(val_float, unit_str, fallback_unit=target_press)
                si_unit = 'bara'
                unit_str = resolved_u if converted else f"{resolved_u} (Override)"
            elif 'diameter' in p_lower:
                si_val, converted, resolved_u = convert_to_si_base(val_float, unit_str, LENGTH_TO_M, fallback_unit=target_diameter)
                si_unit = 'm'
                unit_str = resolved_u if converted else f"{resolved_u} (Override)"
        except (ValueError, TypeError):
            pass
            
        rows_out.append({
            'Parameter': param_str,
            'Value': val_float,
            'Units': unit_str,
            'Internal_SI_Val': si_val,
            'Internal_SI_Unit': si_unit
        })
    return pd.DataFrame(rows_out)

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
    lookup = {}
    for _, row in prop_df.iterrows():
        p_norm = row['Parameter'].strip().lower()
        if 'pressure' in p_norm:
            lookup['p_bara'] = row['Internal_SI_Val']
        elif 'temperature' in p_norm:
            lookup['t_c'] = row['Internal_SI_Val']
        elif 'weight' in p_norm or 'mw' in p_norm:
            lookup['mw'] = row['Internal_SI_Val']
        elif 'compressibility' in p_norm or 'z' in p_norm:
            lookup['z'] = row['Internal_SI_Val']
            
    try:
        return {
            'p_bara': float(lookup['p_bara']),
            't_c': float(lookup['t_c']),
            'mw': float(lookup['mw']),
            'z': float(lookup['z']),
        }
    except (KeyError, TypeError, ValueError):
        return None

def compute_missing_parameter_si(available_si, common_flow_m3hr, gas_props):
    """Performs dynamic missing parameter tracking inside standard SI metrics boundaries."""
    have = set(available_si.keys())
    needed = {'Head', 'Efficiency', 'Power'} - have
    if len(needed) != 1 or gas_props is None:
        return None, None
    missing = needed.pop()

    rho = gas_density_kg_m3(gas_props['p_bara'], gas_props['t_c'], gas_props['mw'], gas_props['z'])
    mass_flow_kg_s = common_flow_m3hr * rho / 3600.0

    head_m = available_si.get('Head', 0.0)
    power_kw = available_si.get('Power', 0.0)
    eff_pct = available_si.get('Efficiency', 0.0)

    if missing == 'Power':
        eff_frac = eff_pct / 100.0
        calc_kw = mass_flow_kg_s * G * head_m / eff_frac / 1000.0
        return 'Power', calc_kw

    if missing == 'Head':
        eff_frac = eff_pct / 100.0
        calc_m = power_kw * 1000.0 * eff_frac / (mass_flow_kg_s * G)
        return 'Head', calc_m

    if missing == 'Efficiency':
        calc_pct = (mass_flow_kg_s * G * head_m) / (power_kw * 1000.0) * 100.0
        return 'Efficiency', calc_pct

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
                st.subheader('Operating Conditions (Processed View)')
                
                # Visual UI presentation mapping to requested dropdown layouts
                display_df = prop_df.copy()
                for idx, row in display_df.iterrows():
                    p_lower = row['Parameter'].lower()
                    if 'temperature' in p_lower:
                        display_df.at[idx, 'Value'] = route_temperature_from_c(row['Internal_SI_Val'], target_temp)
                        display_df.at[idx, 'Units'] = target_temp
                    elif 'pressure' in p_lower:
                        display_df.at[idx, 'Value'] = route_pressure_from_bara(row['Internal_SI_Val'], target_press)
                        display_df.at[idx, 'Units'] = target_press
                    elif 'diameter' in p_lower:
                        display_df.at[idx, 'Value'] = convert_from_si_base(row['Internal_SI_Val'], target_diameter, LENGTH_TO_M)
                        display_df.at[idx, 'Units'] = target_diameter
                
                st.dataframe(display_df[['Parameter', 'Value', 'Units']], use_container_width=True)
                
                for _, row in display_df.iterrows():
                    property_rows.append([stage, row['Parameter'], row['Value'], row['Units']])
                gas_props = gas_properties_from_df(prop_df)
            else:
                st.warning(f'No operating-conditions block found in {stage}')

            if not blocks:
                st.warning(f'No blocks found in {stage}')
                continue

            stage_models = {}
            stage_parameters = []
            tabs = st.tabs([b['parameter'] for b in blocks])

            for tab, block in zip(tabs, blocks):
                with tab:
                    param = block['parameter']
                    df, flow_conv, val_conv, used_flow_u, used_val_u = extract_block_data(raw, block)
                    stage_parameters.append(param)

                    # Dynamic UI fallback validation notice messages
                    if not flow_conv:
                        st.caption(f"⚠️ Unrecognized Flow unit '{block['flow_unit']}'. Defaulting to Sidebar option: **{target_flow}**.")
                    if not val_conv:
                        st.caption(f"⚠️ Unrecognized {param} unit '{block['value_unit']}'. Defaulting to Sidebar option.")

                    # Convert baseline SI data vectors to target configuration for interactive plots
                    df_plot = df.copy()
                    df_plot['Flow'] = convert_from_si_base(df['Flow'], target_flow, FLOW_TO_M3HR)
                    
                    val_table = {'Head': HEAD_TO_M, 'Power': POWER_TO_KW, 'Efficiency': EFF_TO_PCT}.get(param, {})
                    target_u = {'Head': target_head, 'Power': target_power, 'Efficiency': target_eff}.get(param, '')
                    df_plot['Value'] = convert_from_si_base(df['Value'], target_u, val_table)

                    st.caption(f"Visualizing Data -> Flow: **{target_flow}** | {param}: **{target_u}**")

                    fig = go.Figure()
                    if param not in stage_models:
                        stage_models[param] = {}

                    for speed in sorted(df_plot['Speed'].unique()):
                        sdf = df_plot[df_plot['Speed'] == speed]
                        if len(sdf) < 4:
                            continue

                        x_plot = sdf['Flow'].values.astype(float)
                        y_plot = sdf['Value'].values.astype(float)

                        # Generate regressions using normalized metrics to secure fitting integrity
                        sdf_si = df[df['Speed'] == speed]
                        x_si = sdf_si['Flow'].values.astype(float)
                        y_si = sdf_si['Value'].values.astype(float)

                        if method == 'Auto Best Fit':
                            used, mdl = auto_best(x_si, y_si)
                        else:
                            mdl = build_model(x_si, y_si, method)
                            used = method

                        stage_models[param][speed] = mdl
                        r2_rows.append([stage, speed, param, used, round(mdl['r2'], 6)])

                        # Plotting predictions mapped into targeted dimensions
                        flow_fit_si = np.linspace(x_si.min(), x_si.max(), points)
                        y_fit_si = predict_model(mdl, flow_fit_si)
                        
                        flow_fit_plot = convert_from_si_base(flow_fit_si, target_flow, FLOW_TO_M3HR)
                        y_fit_plot = convert_from_si_base(y_fit_si, target_u, val_table)

                        fig.add_trace(go.Scatter(x=x_plot, y=y_plot, mode='markers', name=f'{speed} Original'))
                        fig.add_trace(go.Scatter(x=flow_fit_plot, y=y_fit_plot, mode='lines', name=f'{speed} Fit'))

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

                common_min_si = max(m['xmin'] for m in available)
                common_max_si = min(m['xmax'] for m in available)

                if common_max_si <= common_min_si:
                    continue

                common_flow_si = np.linspace(common_min_si, common_max_si, points)
                common_flow_target = convert_from_si_base(common_flow_si, target_flow, FLOW_TO_M3HR)
                
                temp = {'Speed': [speed] * points, f'Flow ({target_flow})': common_flow_target}

                predicted_si = {}
                for p in stage_models:
                    if speed in stage_models[p]:
                        vals_si = predict_model(stage_models[p][speed], common_flow_si)
                        predicted_si[p] = vals_si
                        
                        val_table = {'Head': HEAD_TO_M, 'Power': POWER_TO_KW, 'Efficiency': EFF_TO_PCT}.get(p, {})
                        lbl = {'Head': target_head, 'Power': target_power, 'Efficiency': target_eff}.get(p, '')
                        temp[f'{p} ({lbl})'] = convert_from_si_base(vals_si, lbl, val_table)

                if gas_props is not None:
                    name, values_si = compute_missing_parameter_si(predicted_si, common_flow_si, gas_props)
                    if name is not None:
                        computed_param_name = name
                        val_table = {'Head': HEAD_TO_M, 'Power': POWER_TO_KW, 'Efficiency': EFF_TO_PCT}.get(name, {})
                        lbl = {'Head': target_head, 'Power': target_power, 'Efficiency': target_eff}.get(name, '')
                        temp[f'{name} ({lbl}, calculated)'] = convert_from_si_base(values_si, lbl, val_table)

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
    output_filename = f"{input_filename}_Regression_Output_{timestamp}.xlsx"

    st.download_button(
        label="Download Regressed Workbook",
        data=output.getvalue(),
        file_name=output_filename,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
